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
    assert name == "recall"

    with open(path / "files.dep", "a") as f:
        f.write(inspect.getfile(f"{path_to_kwargs}\n"))

    # assumption: we have the embedding generated in the direct parent
    # and the dataset is defined on directory above.  This should
    # maybe have some code that actually searches the parent dirs, but
    # so far this holds for all approaches.
    embeddings_dir = path.parent
    data_dir = path.parent.parent

    X = np.load(embeddings_dir / "1.zip")["embedding"]
    A = sparse.load_npz(data_dir / "1.zip")

    score = graph_knn_recall(X, A, **kwargs)

    npz = np.load(embeddings_dir / "1.zip")
    other_embks = [k for k in npz.keys() if k.startswith("embeddings/step-")]
    n = len("embeddings/step-")
    step_keys = [int(k[n:]) for k in other_embks]
    scores = [graph_knn_recall(npz[k], A, **kwargs) for k in other_embks]
    df_scores = pl.DataFrame(dict(step=step_keys, score=scores))
    with zipfile.ZipFile(embeddings_dir / "1.zip") as zf:
        if "lightning_logs/steps.csv" in zf.namelist():
            with zf.open("lightning_logs/steps.csv") as f:
                df_epochs = pl.read_csv(f)
        else:
            df_epochs = pl.DataFrame(
                dict(global_step=step_keys, epoch=step_keys)
            )

    df = df_scores.join(df_epochs, left_on="step", right_on="global_step")

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}".encode())

        with zf.open("scores.csv", "w") as f:
            df.write_csv(f)


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
