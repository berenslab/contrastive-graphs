import inspect
import zipfile

import numpy as np
import polars as pl
from scipy import sparse
from sklearn import metrics

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "lpred"

    eval_all_embeddings = kwargs.pop("all", False)

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

    score = link_pred(X, A, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}\n".encode())

    if eval_all_embeddings:
        df = lpred_other_embeddings(embeddings_dir, A, **kwargs)
        with zipfile.ZipFile(outfile, "a") as zf:
            with zf.open("scores.csv", "w") as f:
                df.write_csv(f)


def link_pred(
    Z,
    A,
    test_size=1000,
    metric="euclidean",
    mode="auc",
    n_jobs=-1,
    random_state=110099,
):
    Au = sparse.triu(A, k=1).tocoo().astype("int8")
    test_size = min(Au.nnz, test_size)

    rng = np.random.default_rng(random_state)
    test_ind = rng.choice(
        Au.nnz,
        size=test_size,
        replace=False,
    )

    Aneg = sparse.csr_array(Au.shape, dtype=Au.dtype)
    n = len(Z)
    while Aneg.nnz < test_size:
        r, c = rng.integers(n, size=(2, test_size))
        # input to sparse matrix
        data = np.ones(len(r), dtype=Au.dtype), (r, c)
        A1 = sparse.csr_matrix(data, shape=Au.shape)
        Aneg += A1
        Aneg = sparse.triu(((Aneg - Au) > 0).astype("int8"), k=1)
    _A = Aneg.tocoo()
    Aneg = sparse.coo_array(
        (_A.data[:test_size], (_A.row[:test_size], _A.col[:test_size])),
        shape=Aneg.shape,
    )

    rows = np.concat((Au.row[test_ind], Aneg.row))
    cols = np.concat((Au.col[test_ind], Aneg.row))
    y = [1 for _ in range(test_size)] + [0 for _ in range(test_size)]

    dists = score_edges(rows, cols, Z, metric=metric)
    if mode == "auc":
        score = metrics.roc_auc_score(y, dists)
    elif mode == "ap":
        score = metrics.average_precision_score(y, dists)
    else:
        raise ValueError(f"Unkown {mode=!r}, only 'auc', 'ap' allowed")

    return score


def score_edges(rows, cols, embd, metric="cosine"):
    if metric == "cosine":
        norms = np.linalg.norm(embd, axis=1)
        embd = embd / norms[:, None]
        scores = (embd[rows] * embd[cols]).sum(-1)
    elif metric == "euclidean":
        scores = -np.linalg.norm(embd[rows] - embd[cols], axis=1)
    else:
        raise ValueError(f"Unknown {metric=!r}")
    return scores


def lpred_other_embeddings(embeddings_dir, A, **kwargs):
    npz = np.load(embeddings_dir / "1.zip")
    other_embks = [k for k in npz.keys() if k.startswith("embeddings/step-")]
    n = len("embeddings/step-")
    step_keys = [int(k[n:]) for k in other_embks]
    scores = [link_pred(npz[k], A, **kwargs) for k in other_embks]
    df_scores = pl.DataFrame(dict(step=step_keys, score=scores))
    with zipfile.ZipFile(embeddings_dir / "1.zip") as zf:
        if "lightning_logs/steps.csv" in zf.namelist():
            with zf.open("lightning_logs/steps.csv") as f:
                df_epochs = pl.read_csv(f)
        else:
            df_epochs = pl.DataFrame(
                dict(global_step=step_keys, epoch=step_keys)
            )

    return df_scores.join(df_epochs, left_on="step", right_on="global_step")
