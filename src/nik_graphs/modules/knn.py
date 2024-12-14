import zipfile

import numpy as np
from sklearn import neighbors

from ..path_utils import path_to_kwargs


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "knn"

    # assumption: we have the embedding generated in the direct parent
    # and the dataset is defined on directory above.  This should
    # maybe have some code that actually searches the parent dirs, but
    # so far this holds for all approaches.
    embeddings_dir = path.parent
    data_dir = path.parent.parent

    X = np.load(embeddings_dir / "1.zip")["embedding"]
    npz = np.load(data_dir / "1.zip")
    labels = npz["labels"]

    keys = ["train", "test", "val"]
    indd = {f"{k}_inds": npz[f"split/{k}"] for k in keys}

    score = knn(X, labels, **indd, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}".encode())


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
