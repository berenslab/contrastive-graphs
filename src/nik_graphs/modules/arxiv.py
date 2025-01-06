import contextlib
import inspect
import io
import os
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

from ..graph_utils import save_dataset_split, save_graph
from ..path_utils import path_to_kwargs


def run_path(p, outfile):

    name, kwargs = path_to_kwargs(p)
    assert name == "arxiv"

    return ogb_dataset("arxiv", p, outfile)


def ogb_dataset(dataset_key, p, outfile):
    name, kwargs = path_to_kwargs(p)
    random_state = kwargs.pop("random_state", 47**8)
    rng = np.random.default_rng(random_state)
    assert kwargs == dict()

    cache_dir = os.environ.get("XDG_CACHE_DIR", Path.home() / ".cache")
    data_dir = Path(cache_dir) / "open_graph_benchmark"

    dgl_output = io.StringIO()
    with contextlib.redirect_stdout(dgl_output):
        dataset = DglNodePropPredDataset(f"ogbn-{dataset_key}", root=data_dir)

    g, labels = dataset[0]
    if dataset_key != "mag":
        labels = labels.squeeze()
        features = g.ndata["feat"].numpy()
        G = g.to_networkx().to_undirected()
    else:
        labels = labels["paper"].numpy().squeeze()
        features = g.ndata["feat"]["paper"].numpy()
        G = g[("paper", "cites", "paper")].to_networkx().to_undirected()

    # preprocess
    G.remove_edges_from(nx.selfloop_edges(G))

    # Isolate the largest connected component
    sel = list(sorted(nx.connected_components(G), key=len, reverse=True)[0])

    G = G.subgraph(sel)
    labels = labels[sel]
    features = features[sel, :]
    A = nx.adjacency_matrix(G).astype("uint8")

    save_graph(
        outfile,
        A,
        features,
        labels,
        save_spectral=True,
        random_state=rng.integers(2**31 - 1),
    )

    m = "train_mask"
    has_split = m in g.ndata and g.ndata[m] != dict()
    if not has_split:
        n = A.shape[0]
        train_size = n * 8 // 10
        test_size = n // 10
        # val_size = n // 10
        inds = rng.permutation(A.shape[0]).astype("uint32")
        train_inds = inds[:train_size]
        test_inds = inds[train_size : train_size + test_size]
        val_inds = inds[train_size + test_size :]
    else:

        def mask2ind(x):
            return np.where(x)[0]

        train_inds = mask2ind(g.ndata["train_mask"])
        test_inds = mask2ind(g.ndata["test_mask"])
        val_inds = mask2ind(g.ndata["val_mask"])
    save_dataset_split(outfile, train_inds, test_inds, val_inds)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("dgl_class_info.txt", "w") as f:
            f.write(dgl_output.getvalue().encode())
    with open(p / "files.dep", "a") as f:
        pyobjs = [save_graph, path_to_kwargs]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]
        f.write(f"{__file__}\n")
