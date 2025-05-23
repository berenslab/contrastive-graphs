import warnings
import zipfile

import GCL.augmentors as A
import GCL.losses as L
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from GCL.models import DualBranchContrast
from scipy import sparse
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from ..path_utils import path_to_kwargs

__partition__ = "2080-galvani"


def run_path(path, outfile):
    torch.set_float32_matmul_precision("medium")
    zipf = path.parent / "1.zip"

    name, kwargs = path_to_kwargs(path)
    assert name == "gcl"

    adj = sparse.load_npz(zipf).tocoo()
    npz = np.load(zipf)
    features = npz["features"]

    with warnings.catch_warnings():
        msg = "'dropout_adj' is deprecated, use 'dropout_edge' instead"
        warnings.filterwarnings(
            action="ignore", category=UserWarning, message=msg
        )

        Y = graphcl(adj, features, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def graphcl(adj, feat, batch_size=128, device="cuda:0", **kwargs):

    graph = torch_geometric.data.Data(
        x=torch.from_numpy(feat),
        edge_index=torch.tensor(np.asarray([adj.row, adj.col])),
    )
    dataloader = DataLoader(ListDataset([graph]), batch_size=batch_size)
    # dataloader = DataLoader(np.array([graph]), batch_size=batch_size)

    return graphcl_dataloader(
        dataloader,
        input_dim=feat.shape[1],
        device=device,
        # batch_size=batch_size,
        **kwargs,
    )


def graphcl_dataloader(
    dataloader,
    input_dim,
    device,
    n_epochs=100,
    lr=0.01,
    hidden_dim=32,
    num_layers=2,
    out_dim=2,
    temp=0.2,
    mode="G2G",
):
    aug1 = A.Identity()
    aug2 = A.RandomChoice(
        [
            A.RWSampling(num_seeds=1000, walk_length=10),
            A.NodeDropping(pn=0.1),
            A.FeatureMasking(pf=0.1),
            A.EdgeRemoving(pe=0.1),
        ],
        1,
    )
    gconv = GConv(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=out_dim,
    ).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(
        loss=L.InfoNCE(tau=temp), mode=mode
    ).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        _ = train(encoder_model, contrast_model, dataloader, optimizer)

    return transform(encoder_model, dataloader, device=device)


class ListDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, datalist):
        super().__init__(
            root=None, transform=None, pre_transform=None, pre_filter=None
        )
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def get(self, idx):
        return self.datalist[idx]


def make_gin_conv(input_dim, out_dim):
    return GINConv(
        nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
    )


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, out_dim):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, out_dim),
        )

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to("cuda")
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones(
                (num_nodes, 1), dtype=torch.float32, device=data.batch.device
            )

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def transform(encoder_model, dataloader, device="cuda"):
    encoder_model.eval()
    x = []
    for data in dataloader:
        data = data.to(device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones(
                (num_nodes, 1), dtype=torch.float32, device=data.batch.device
            )
        _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(encoder_model.encoder.project(g))
    x = torch.cat(x, dim=0).cpu().float().detach().numpy()
    return x
