import contextlib
import inspect
import io
import os
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
from dgl.data import CoraGraphDataset

from ..graph_utils import (
    save_dataset_split,
    save_graph,
    save_public_dataset_split,
)
from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(p, outfile):

    name, kwargs = path_to_kwargs(p)
    assert name == "cora"
    cls = CoraGraphDataset
    return dgl_dataset(cls, p, outfile)


def dgl_dataset(cls, p, outfile):
    name, kwargs = path_to_kwargs(p)
    random_state = kwargs.pop("random_state", 47**8)
    rng = np.random.default_rng(random_state)
    assert kwargs == dict()

    cache_dir = os.environ.get("XDG_CACHE_DIR", Path.home() / ".cache")
    data_dir = Path(cache_dir) / "deep_graph_library"

    dgl_output = io.StringIO()
    with contextlib.redirect_stdout(dgl_output):
        g = cls(data_dir)[0]
    labels = g.ndata["label"].numpy()
    features = g.ndata["feat"].numpy()
    G = g.to_networkx().to_undirected()

    # preprocess
    G.remove_edges_from(nx.selfloop_edges(G))

    # Isolate the largest connected component
    sel = list(sorted(nx.connected_components(G), key=len, reverse=True)[0])

    G = G.subgraph(sel)
    G = nx.relabel_nodes(G, mapping={g: i for i, g in enumerate(G.nodes)})
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

    n = A.shape[0]
    train_size = n * 8 // 10
    test_size = n // 10
    # val_size = n // 10
    inds = rng.permutation(A.shape[0]).astype("uint32")
    train_inds = inds[:train_size]
    test_inds = inds[train_size : train_size + test_size]
    val_inds = inds[train_size + test_size :]
    save_dataset_split(outfile, train_inds, test_inds, val_inds)

    if "train_mask" in g.ndata:

        def key2ind(k):
            return g.ndata[k][sel].nonzero()[0]

        train_inds = key2ind("train_mask")
        test_inds = key2ind("test_mask")
        val_inds = key2ind("val_mask")
        save_public_dataset_split(outfile, train_inds, test_inds, val_inds)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("dgl_class_info.txt", "w") as f:
            f.write(dgl_output.getvalue().encode())
    with open(p / "files.dep", "a") as f:
        pyobjs = [save_graph, path_to_kwargs]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]
        f.write(__file__ + "\n")
