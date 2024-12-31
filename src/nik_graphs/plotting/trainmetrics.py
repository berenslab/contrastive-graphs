# example plotname = "trainmetrics"
def deplist(plotname):
    assert plotname.name == "trainmetrics"

    return ["../dataframes/learntemp_evals.parquet"]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist(plotname)[0])

    return plot(
        df,
        outfile=outfile,
        format=format,
    )


def plot(df, outfile=None, format="pdf"):
    from matplotlib import pyplot as plt

    from ..plot import add_letters

    datasets = df["dataset"].unique()

    # figsize is 3.25 inches, that is a single column in icml 2025 paper format.
    # figsize=(3.25, 1.1),
    fig = plt.figure(figsize=(3.25, 1 * len(datasets)))
    figs = fig.subfigures(len(datasets), 1, squeeze=False)

    for sfig, ((dataset,), df_) in zip(
        figs.flat, df.group_by("dataset", maintain_order=True)
    ):
        plot_dataset(sfig, df_, dataset)
        sfig.suptitle(dataset)
    add_letters(fig.get_axes())
    [ax.set_xlabel("epoch") for fig in figs[-1] for ax in fig.get_axes()]
    (ax,) = [ax for ax in figs[0, 0].get_axes() if ax.get_label() == "acc"]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        ncols=3,
        fontsize="small",
        columnspacing=1.25,
    )
    fig.savefig(outfile, format=format)


def plot_dataset(fig, df, dataset_name):
    import matplotlib as mpl
    import polars as pl

    axd = fig.subplot_mosaic([["loss", "temperature", "acc"]])

    dfg = df.group_by("epoch")
    batches_per_epoch = df["batches_per_epoch"][0]

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

    [ax.sharex(axd["loss"]) for k, ax in axd.items() if k != "loss"]
    [set_bounds(ax) for k, ax in axd.items()]

    ax = axd["temperature"]
    epoch, logtemp = dfg.agg(pl.mean("logtemp")).sort(by="epoch")[
        ["epoch", "logtemp"]
    ]
    temp = logtemp.exp()
    (line,) = ax.plot(epoch, temp, label="temperature", color="xkcd:dark grey")
    plot_steps(ax, "logtemp", color=line.get_color())
    ax.set(ylabel=r"temperature $\tau$", yscale="log")
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
    ax.set_ylabel("loss")

    ax = axd["acc"]
    for k in ["knn", "lin", "recall"]:
        epoch, score = df[["epoch", k]].drop_nulls()
        ax.plot(epoch, score, label=k)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
