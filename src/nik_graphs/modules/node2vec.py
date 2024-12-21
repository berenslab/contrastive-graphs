import sys
import zipfile
from pathlib import Path

import lightning
import numpy as np
import torch
import torch_geometric
from dgl import from_scipy
from dgl.nn import DeepWalk
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from ..path_utils import path_to_kwargs

# from torch_geometric.nn import Node2Vec
__partition__ = "2080-galvani"


def run_path(path, outfile):
    torch.set_float32_matmul_precision("medium")
    zipf = path.parent / "1.zip"

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "node2vec"
    Y = node2vec(A, verbose=False)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def node2vec(
    A,
    dim,
    batch_size=128,
    lr=0.01,
    n_epochs=100,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=1.0,
    q=1.0,
    verbose=False,
    **kwargs,
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    edge_index = from_scipy_sparse_matrix(A).edge_index

    model = torch_geometric.nn.Node2Vec(
        edge_index,
        embedding_dim=dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=1)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(1, n_epochs + 1):
        loss = train()
        if verbose:
            print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}", file=sys.stderr)

    Z = model().detach().cpu().numpy()
    return Z
