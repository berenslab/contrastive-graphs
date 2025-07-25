import inspect
import zipfile

import networkx as nx
import numpy as np
import polars as pl
from scipy import sparse
from sklearn import preprocessing

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "spcorr"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n")

    # assumption: we have the embedding generated in the direct parent
    # and the dataset is defined on directory above.  This should
    # maybe have some code that actually searches the parent dirs, but
    # so far this holds for all approaches, except for the case where
    # we evaluate the feature space of the dataset directly.
    embeddings_dir = path.parent
    data_dir = path.parent.parent
    # if we evaluate the input feature space directly, we change the
    # data_dir and the corresponding embedding key
    if data_dir.name == "runs":
        data_dir = embeddings_dir
        embedding_key = "features"
    else:
        embedding_key = "embedding"

    X = np.load(embeddings_dir / "1.zip")[embedding_key]
    A = sparse.load_npz(data_dir / "1.zip")
    G = nx.from_scipy_sparse_array(A)

    score = shortest_path_corr(X, G, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}\n".encode())


def shortest_path_corr(
    Z, G, test_size=1000, metric="euclidean", n_jobs=-1, random_state=101049
):
    test_size = min(Z.shape[0], test_size)
    src_ind, tgt_ind = test_ind = np.random.default_rng(random_state).choice(
        Z.shape[0],
        size=(2, test_size),
        replace=False,
    )
    graph_dists = [
        nx.astar_path_length(G, src, tgt) for src, tgt in test_ind.T
    ]
    if metric == "euclidean":
        emb_dists = ((Z[src_ind] - Z[tgt_ind]) ** 2).sum(axis=1) ** 0.5
    elif metric == "cosine":
        Z_norm = preprocessing.normalize(Z)
        emb_dists = ((Z_norm[src_ind] - Z_norm[tgt_ind]) ** 2).sum(
            axis=1
        ) ** 0.5

    return np.corrcoef(graph_dists, emb_dists)[0, 1]
