import inspect
import os
from pathlib import Path

import numpy as np
from sklearn import datasets, decomposition

from ..graph_utils import make_adj_mat, save_dataset_split, save_graph
from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(p, outfile):

    name, kwargs = path_to_kwargs(p)
    assert name == "mnist"
    random_state = kwargs.pop("random_state", 47**8)
    rng = np.random.default_rng(random_state)
    assert kwargs == dict()

    cache_dir = os.environ.get("XDG_CACHE_DIR", Path.home() / ".cache")
    data_dir = Path(cache_dir) / "scikit_learn_data"
    mnist = datasets.fetch_openml("mnist_784", data_home=data_dir)

    pca = decomposition.PCA(
        50, copy=False, random_state=rng.integers(2**32 - 1)
    )
    X = pca.fit_transform(mnist["data"].values)

    adj, annoy_index = make_adj_mat(X, seed=rng.integers(2**32 - 1))
    adj = adj.tocsr().astype("uint8")
    labels = mnist["target"].cat.codes.values
    # will not use `G` here because it's derived from the adjacency matrix
    # G = nx.from_scipy_sparse_array(adj).to_undirected()

    assert not Path(outfile).exists(), f"{outfile} must not exist."
    save_graph(outfile, adj, X.astype("float32"), labels)

    train_size = 50_000
    test_size = 10_000
    # val_size = 10_000
    inds = rng.permutation(X.shape[0]).astype("uint32")
    train_inds = inds[:train_size]
    test_inds = inds[train_size : train_size + test_size]
    val_inds = inds[train_size + test_size :]
    save_dataset_split(outfile, train_inds, test_inds, val_inds)

    with open(p / "files.dep", "a") as f:
        pyobjs = [path_to_kwargs, save_graph]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]
