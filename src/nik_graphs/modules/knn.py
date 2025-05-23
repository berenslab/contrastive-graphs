import inspect
import zipfile

import numpy as np
import polars as pl
from sklearn import neighbors

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "knn"

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
    npz = np.load(data_dir / "1.zip")
    labels = npz["labels"]
    keys = ["train", "test", "val"]
    indd = {f"{k}_inds": npz[f"split/{k}"] for k in keys}

    score = knn(X, labels, **indd, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}\n".encode())

    if eval_all_embeddings:
        df = knn_other_embeddings(embeddings_dir, labels, **indd, **kwargs)
        with zipfile.ZipFile(outfile, "a") as zf:
            with zf.open("scores.csv", "w") as f:
                df.write_csv(f)


def knn(
    X,
    labels,
    train_inds,
    test_inds,
    val_inds=None,
    k=15,
    metric="euclidean",
    n_jobs=-1,
):
    knn = neighbors.KNeighborsClassifier(k, metric=metric, n_jobs=n_jobs)

    if val_inds is not None:
        train_inds = np.hstack((train_inds, val_inds))

    X_train = X[train_inds]
    X_test = X[test_inds]
    y_train = labels[train_inds]
    y_test = labels[test_inds]

    return knn.fit(X_train, y_train).score(X_test, y_test)


def knn_other_embeddings(embeddings_dir, labels, **kwargs):
    npz = np.load(embeddings_dir / "1.zip")
    other_embks = [k for k in npz.keys() if k.startswith("embeddings/step-")]
    n = len("embeddings/step-")
    step_keys = [int(k[n:]) for k in other_embks]
    scores = [knn(npz[k], labels, **kwargs) for k in other_embks]
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
