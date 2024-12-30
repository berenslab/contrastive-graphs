import zipfile
from pathlib import Path


# example plotname = "mnist.trainmetrics". The middle part must not
# contain dots as part of the arguments.
def deplist(plotname):
    return list(deps(plotname).values())


def deps(plotname):
    dataset, _name = plotname.name.split(".")
    assert _name == "trainmetrics"

    path = Path("../runs") / dataset / "cne,n_epochs=5"
    depdict = {k: path / k / "1.zip" for k in ["lin", "knn", "recall"]}
    depdict["embedding"] = path / "1.zip"
    depdict["data"] = path.parent / "1.zip"
    return depdict


def plot_path(plotname, outfile, format="pdf"):
    import numpy as np
    import polars as pl

    depdict = deps(plotname)

    # embedding = np.load(files["embedding"])["embedding"]
    # if embedding.shape[1] > 2:
    #     raise RuntimeError(
    #         f"Need a 2D embedding, but given array is {embedding.shape[1]}D"
    #     )

    zpath = zipfile.Path(depdict["embedding"])
    with (zpath / "lightning_logs/metrics.csv").open() as f:
        train_df = pl.read_csv(f)

    # labels = np.load(files["data"])["labels"]
    df_dict = dict()
    df_ = None
    for k in ["lin", "knn", "recall"]:
        zipf = depdict[k]
        with zipfile.ZipFile(zipf) as zf:
            with zf.open("scores.csv") as f:
                df1 = pl.read_csv(f).drop("step")

    df__ = df1.with_columns().df__.rename(dict(score=k), strict=False)
    df_ = df_.vstack(df__) if df_ is not None else df__
    df_dict[k] = df_
    df = pl.concat(
        [df_dict["knn"].drop("knn")] + [df[k] for k, df in df_dict.items()],
        how="horizontal",
    )

    return plot(train_df, df, outfile=outfile, format=format)


def plot(train_df, df, outfile=None, format="pdf"):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.scatter(*embedding.T, c=labels, alpha=0.6, rasterized=True)
    ax.set_aspect(1)

    txt = "\n".join(f"{k} = {v:.1%}" for k, v in accd.items())
    ax.text(
        1,
        1,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        ma="right",
        family="monospace",
    )

    fig.savefig(outfile, format=format)
