import zipfile
from pathlib import Path


# example plotname = "mnist.trainmetrics". The middle part must not
# contain dots as part of the arguments.
def deplist(plotname):
    return list(deps(plotname).values())


def deps(plotname):
    dataset, _name = plotname.name.split(".")
    assert _name == "trainmetrics"

    path = Path("../runs") / dataset / "cne,loss=infonce-temp,n_epochs=5"
    depdict = {k: path / k / "1.zip" for k in ["lin", "knn", "recall"]}
    depdict["embedding"] = path / "1.zip"
    depdict["data"] = path.parent / "1.zip"
    return depdict


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl
    import yaml

    depdict = deps(plotname)

    zpath = zipfile.Path(depdict["embedding"])
    with (zpath / "lightning_logs/metrics.csv").open() as f:
        train_df = pl.read_csv(f)
    with (zpath / "lightning_logs/hparams.yaml").open() as f:
        hparams = yaml.safe_load(f)

    # labels = np.load(files["data"])["labels"]
    df_dict = dict()
    for k in ["lin", "knn", "recall"]:
        zipf = depdict[k]
        with zipfile.ZipFile(zipf) as zf:
            with zf.open("scores.csv") as f:
                df1 = pl.read_csv(f)

        df_ = df1.rename(dict(score=k), strict=False)
        df_dict[k] = df_
    df = pl.concat(
        [df_dict["knn"].drop("knn")]
        + [pl.DataFrame(df[k]) for k, df in df_dict.items()],
        how="horizontal",
    )

    return plot(
        train_df,
        df,
        batches_per_epoch=hparams["batches_per_epoch"],
        outfile=outfile,
        format=format,
    )


def plot(train_df, df, batches_per_epoch=None, outfile=None, format="pdf"):
    import matplotlib as mpl
    import polars as pl
    from matplotlib import pyplot as plt

    dfg = train_df.group_by("epoch")

    def plot_steps(ax, k, color):
        if batches_per_epoch is not None:
            df = train_df[["step", k]].drop_nulls()
            s = df[k] if k != "logtemp" else df[k].exp()
            ax.plot(
                df["step"] / batches_per_epoch,
                s,
                color=color,
                alpha=0.618,
            )
        else:
            import warnings

            warnings.warn(
                f"{batches_per_epoch=} so only mean over epoch will be plotted"
            )

    def set_bounds(ax):
        if batches_per_epoch is not None:
            ax.spines.bottom.set_bounds(
                0, (train_df["step"] / batches_per_epoch).ceil().max()
            )

    # figsize is 3.25 inches, that is a single column in icml 2025 paper format.
    fig, axd = plt.subplot_mosaic(
        [["plot", "acc"], ["temperature", "legend"]],
        figsize=(3.25, 2),
    )
    [ax.sharex(axd["plot"]) for k, ax in axd.items() if k != "plot"]
    [set_bounds(ax) for k, ax in axd.items()]

    ax = axd["temperature"]
    epoch, logtemp = dfg.agg(pl.mean("logtemp")).sort(by="epoch")[
        ["epoch", "logtemp"]
    ]
    temp = logtemp.exp()
    (line,) = ax.plot(epoch, temp, label="temperature", color="xkcd:dark grey")
    plot_steps(ax, "logtemp", color=line.get_color())
    ax.set(xlabel="epoch", ylabel="temperature", yscale="log")

    ax = axd["plot"]
    for k in ["loss"]:
        epoch, val = dfg.agg(pl.mean(k)).sort(by="epoch")[["epoch", k]]
        (line,) = ax.plot(epoch, val, label=k)
        color = line.get_color()
        plot_steps(ax, k, color)

    ax.legend()

    ax = axd["acc"]
    for k in ["knn", "lin", "recall"]:
        ax.plot(df["epoch"], df[k], label=k)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.legend()
    ax.set(xlabel="epoch", ylim=(0, 1))

    # handles, labels = ax.get_legend_handles_labels()
    # axd["legend"].legend(handles=handles, labels=labels, loc="center")
    axd["legend"].set_axis_off()

    fig.savefig(outfile, format=format)
