import zipfile
from pathlib import Path


# example plotname = "mnist.trainmetrics". The middle part must not
# contain dots as part of the arguments.
def deplist(plotname):
    return list(deps(plotname).values())


def deps(plotname):
    dataset, _name = plotname.name.split(".")
    assert _name == "trainmetrics"

    path = Path("../runs") / dataset / "cne,loss=infonce-temp"
    depdict = {k: path / k / "1.zip" for k in ["lin", "knn", "recall"]}
    depdict["embedding"] = path / "1.zip"
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

    # subtract 1 from the step so it aligns with the steps in train_df
    df1 = (
        df.select(pl.all(), s=pl.col("step") - 1)
        .drop("step")
        .rename(dict(s="step"))
    )
    df2 = train_df.drop(
        ["dof", "ta", "ru", "val_logtemp", "val_ru", "val_ta", "val_loss"]
    ).drop_nulls()
    dfj = df1.join(df2, on=["epoch", "step"], how="right")

    return plot(
        dfj,
        batches_per_epoch=hparams["batches_per_epoch"],
        outfile=outfile,
        format=format,
    )


def plot(df, batches_per_epoch=None, outfile=None, format="pdf"):
    import matplotlib as mpl
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import add_letters

    dfg = df.group_by("epoch")

    def plot_steps(ax, k, color):
        if batches_per_epoch is not None:
            df_ = df[["step", k]].drop_nulls()
            s = df_[k] if k != "logtemp" else df_[k].exp()
            ax.plot(
                df_["step"] / batches_per_epoch,
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
                0, (df["step"] / batches_per_epoch).ceil().max()
            )

    # figsize is 3.25 inches, that is a single column in icml 2025 paper format.
    fig, axd = plt.subplot_mosaic(
        [["loss", "temperature", "acc"]],
        figsize=(3.25, 1.1),
    )
    [ax.sharex(axd["loss"]) for k, ax in axd.items() if k != "loss"]
    [set_bounds(ax) for k, ax in axd.items()]
    add_letters(axd.values())

    ax = axd["temperature"]
    epoch, logtemp = dfg.agg(pl.mean("logtemp")).sort(by="epoch")[
        ["epoch", "logtemp"]
    ]
    temp = logtemp.exp()
    (line,) = ax.plot(epoch, temp, label="temperature", color="xkcd:dark grey")
    plot_steps(ax, "logtemp", color=line.get_color())
    ax.set(xlabel="epoch", ylabel=r"temperature $\tau$", yscale="log")
    last_temp = temp.drop_nulls()[-1]
    ax.text(
        1,
        last_temp,
        rf"$\tau = {last_temp:.1g}$",
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )

    ax = axd["loss"]
    epoch, val = dfg.agg(pl.mean("loss")).sort(by="epoch")[["epoch", "loss"]]
    (line,) = ax.plot(epoch, val, label="train", color="xkcd:dark grey")
    # epoch, val = (
    #     dfg.agg(pl.mean("val_loss"))
    #     .sort(by="epoch")[["epoch", "val_loss"]]
    #     .drop_nulls()
    # )
    # ax.plot(epoch, val, label="val", color=line.get_color())
    plot_steps(ax, "loss", line.get_color())
    ax.set(xlabel="epoch", ylabel="loss")

    ax = axd["acc"]
    for k in ["knn", "lin", "recall"]:
        epoch, score = df[["epoch", k]].drop_nulls()
        ax.plot(epoch, score, label=k)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    ax.legend()
    ax.set(xlabel="epoch")

    fig.savefig(outfile, format=format)
