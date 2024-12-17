import inspect
import zipfile

import numpy as np
import polars as pl
from sklearn import linear_model, pipeline, preprocessing

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "lin"

    with open(path / "files.dep", "a") as f:
        f.write(inspect.getfile(f"{path_to_kwargs}\n"))

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

    score = lin(X, labels, **indd, **kwargs)

    npz = np.load(embeddings_dir / "1.zip")
    other_embks = [k for k in npz.keys() if k.startswith("embeddings/step-")]
    n = len("embeddings/step-")
    step_keys = [int(k[n:]) for k in other_embks]
    scores = [lin(npz[k], labels, **indd, **kwargs) for k in other_embks]
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


def lin(
    X,
    labels,
    train_inds,
    test_inds,
    val_inds=None,
    penalty=None,
    solver="saga",
    max_iter=100,
    tol=0.01,
    n_jobs=-1,
    random_state=54**5,
):
    lin = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            penalty=penalty,
            solver=solver,
            tol=tol,
            n_jobs=-1,
            max_iter=max_iter,
            random_state=random_state,
        ),
    )

    if val_inds is not None:
        train_inds = np.hstack((train_inds, val_inds))

    X_train = X[train_inds]
    X_test = X[test_inds]
    y_train = labels[train_inds]
    y_test = labels[test_inds]

    return lin.fit(X_train, y_train).score(X_test, y_test)
