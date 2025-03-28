import inspect
import zipfile

import dgl
import numpy as np
import torch
from scipy import sparse

from ..path_utils import path_to_kwargs
from .graphmae_extra import (
    build_args,
    build_model,
    create_optimizer,
    pretrain,
    scale_feats,
    set_random_seed,
)

__partition__ = "2080-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "graphmae"

    zipf = path.parent / "1.zip"
    adj = sparse.load_npz(zipf)
    npz = np.load(zipf)
    feat = npz["features"]

    Y = graphmae(adj, feat, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)

    with open(path / "files.dep", "a") as f:
        f.write(inspect.getfile(build_args) + "\n")


def graphmae(
    adj, feat, n_epochs=100, opt="adam", random_state=50123, device="cuda:0"
):

    A = adj.tocoo()

    graph = dgl.DglGraph((A.row, A.col))
    graph = dgl.add_self_loop(graph)

    args = build_args()

    set_random_seed(random_state)

    lr = args.lr
    weight_decay = args.weight_decay

    graph = make_dataset(adj, feat)
    args.num_features = feat.shape[1]

    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(opt, model, lr, weight_decay)
    model = pretrain(
        model, graph, graph.ndata["feat"], optimizer, n_epochs, device=device
    )

    Y = model.embed(graph.to(device), feat.to(device))
    return Y.cpu().detach().numpy()


def make_dataset(adj, feat):
    graph = dgl.graph((adj.row, adj.col))
    graph.ndata["feat"] = torch.from_numpy(scale_feats(feat))

    return graph
