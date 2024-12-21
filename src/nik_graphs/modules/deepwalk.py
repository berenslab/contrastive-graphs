import sys
import zipfile

import numpy as np
import torch
from dgl import from_scipy
from dgl.nn import DeepWalk
from scipy import sparse

from ..path_utils import path_to_kwargs

__partition__ = "2080-galvani"


def run_path(path, outfile):
    torch.set_float32_matmul_precision("medium")
    zipf = path.parent / "1.zip"

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "deepwalk"
    Y = deepwalk(A, verbose=False)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def deepwalk(A, verbose=True, batch_size=128, lr=0.01, n_epochs=100):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    g = from_scipy(A, device=device)
    model = DeepWalk(g, emb_dim=128).to(device)

    loader = torch.utils.data.DataLoader(
        torch.arange(g.num_nodes()),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=model.sample,
    )

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for batch_walk in loader:
            loss = model(batch_walk.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(loader)

        if verbose:
            print(
                f"Epoch: {epoch:03d}, Loss: {total_loss:.4f}", file=sys.stderr
            )

    Z = model.node_embed.weight.detach().cpu().numpy()
    return Z
