import inspect
import zipfile

import numpy as np
from scipy import sparse
from sklearn import neighbors

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "recall"

    with open(path / "files.dep", "a") as f:
        pyobjs = [inspect, zipfile, np, neighbors, path_to_kwargs]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]

    # assumption: we have the embedding generated in the direct parent
    # and the dataset is defined on directory above.  This should
    # maybe have some code that actually searches the parent dirs, but
    # so far this holds for all approaches.
    embeddings_dir = path.parent
    data_dir = path.parent.parent

    X = np.load(embeddings_dir / "1.zip")["embedding"]
    A = sparse.load_npz(data_dir / "1.zip")

    score = graph_knn_recall(X, A, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}".encode())


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
        set1 = set(Z_neighb[i, : A_array[i].sum()])  # neighbors in Z (embed)
        set2 = set(np.where(A_array[i] > 0)[0])  # neighbors in A (true)
        fraction += len(set1 & set2) / len(set2)  # overlap
    fraction /= test_size

    return fraction
