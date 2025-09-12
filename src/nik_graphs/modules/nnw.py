import inspect
import zipfile

import numpy as np
import polars as pl
from scipy import sparse
from sklearn import neighbors

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "nnw"

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

    score = graph_knn_recall(X, A, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}\n".encode())


def graph_knn_recall(
    Z, A, test_size=1000, metric="euclidean", n_jobs=-1, random_state=110099
):
    test_size = min(Z.shape[0], test_size)
    test_ind = np.random.default_rng(random_state).choice(
        Z.shape[0],
        size=test_size,
        replace=False,
    )

    A_array = A[test_ind].toarray().astype(int)
    max_edges = A_array.sum(axis=1).max().astype(int)

    neigh = neighbors.NearestNeighbors(
        n_neighbors=max_edges + 1, metric=metric, n_jobs=n_jobs
    )
    neigh.fit(Z)

    # Excluding point itself as a neighbor
    Z_neighb = neigh.kneighbors(Z[test_ind], return_distance=False)[:, 1:]

    fraction = 0
    for i in range(test_size):
        neighbor_edges = A_array[A_array[i]].sum(1).astype(float)
        neighbor_edges[i] = 0
        neighbor_edges /= neighbor_edges.sum()
        fraction += (A_array[i] * neighbor_edges)[
            Z_neighb[i, : A_array[i].sum()]
        ].mean()
    fraction /= test_size

    return fraction
