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
    import polars as pl

    depdict = deps(plotname)

    zpath = zipfile.Path(depdict["embedding"])
    with (zpath / "lightning_logs/metrics.csv").open() as f:
        train_df = pl.read_csv(f)

    # labels = np.load(files["data"])["labels"]
    df_dict = dict()
    for k in ["lin", "knn", "recall"]:
        zipf = depdict[k]
        with zipfile.ZipFile(zipf) as zf:
            with zf.open("scores.csv") as f:
                df1 = pl.read_csv(f).drop("step")

        df_ = df1.rename(dict(score=k), strict=False)
        df_dict[k] = df_
    df = pl.concat(
        [df_dict["knn"].drop("knn")]
        + [pl.DataFrame(df[k]) for k, df in df_dict.items()],
        how="horizontal",
    )

    return plot(train_df, df, outfile=outfile, format=format)


def plot(train_df, df, outfile=None, format="pdf"):
    import polars as pl
    from matplotlib import pyplot as plt

    dfg = train_df.group_by("epoch")

    # figsize is 3.25 inches, that is a single column in icml 2025 paper format.
    fig, axd = plt.subplot_mosaic(
        [["plot", "legend"]], figsize=(3.25, 2.5), width_ratios=[1, 0.25]
    )
    ax = axd["plot"]
    for k in ["logtemp", "loss"]:
        epoch, val = dfg.agg(pl.mean(k)).sort(by="epoch")[["epoch", k]]
        if k == "logtemp":
            k = "temperature"
            val = val.exp()
        ax.plot(epoch, val, label=k)

    for k in ["knn", "lin", "recall"]:
        ax.plot(df["epoch"], df[k], label=k)

    handles, labels = ax.get_legend_handles_labels()
    axd["legend"].legend(handles=handles, labels=labels, loc="center")
    axd["legend"].set_axis_off()

    fig.savefig(outfile, format=format)
