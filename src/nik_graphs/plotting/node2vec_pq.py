def deplist(plotname=None):
    return ["../dataframes/node2vec_pq.parquet"]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl
    from matplotlib import pyplot as plt

    df = pl.read_parquet(deplist(plotname)[0])

    with plt.rc_context({"xtick.labelsize": 6}):
        fig = plot(df)
        fig.savefig(outfile, format=format)


def plot(df_full):
    import matplotlib as mpl
    import numpy as np
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import translate_plotname  # translate_acc_short,

    metrics = ["recall", "knn", "lin"]

    n_bars = len(df_full["q"].unique())
    bar_width = 1 / (n_bars + 1.62)
    _dftix = df_full.unique("p", maintain_order=True).with_row_index()
    cmap = plt.get_cmap("copper")
    colors = [cmap(x) for x in np.linspace(0.3, 1, num=n_bars)]

    fig, axxs = plt.subplots(3, 9, figsize=(6.75, 3))

    ax_iter = iter(axxs.flat)
    for (dataset,), df in df_full.group_by("dataset", maintain_order=True):

        for m in metrics:
            ax = next(ax_iter)
            if m == "knn":
                ax.set_title(translate_plotname(dataset))
            for i, (((q,), df_), color) in enumerate(
                zip(df.group_by("q", maintain_order=True), colors)
            ):
                x, mean, std = (
                    df_.group_by("p", maintain_order=True)
                    .agg(mean=pl.mean(m), std=pl.std(m))
                    .with_row_index()
                    .select("index", "mean", "std")
                )
                ax.bar(
                    x + i * bar_width,
                    mean,
                    width=bar_width,
                    color=color,
                    label=f"${q:g}$",
                )
                ax.errorbar(
                    x + i * bar_width,
                    mean,
                    yerr=std,
                    fmt="none",
                    ecolor="xkcd:dark grey",
                    zorder=5,
                )

            [ax.axhline(y, color="white") for y in [0.25, 0.5, 0.75]]
            ax.update_datalim([(0, 1)])
            ax.tick_params("both", length=0)
            ax.set(yticks=[], xticks=[])
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            ax.hlines(
                [0] * len(_dftix),
                xmin=_dftix["index"] - bar_width / 2,
                xmax=_dftix["index"] + bar_width * n_bars - bar_width / 2,
                lw=plt.rcParams["axes.linewidth"],
                color="black",
                clip_on=False,
            )

            [ax.spines[x].set_visible(False) for x in ["bottom", "left"]]
            ax.margins(x=0)

    [ax.set_axis_off() for ax in ax_iter]
    handles, labels = ax.get_legend_handles_labels()
    axxs[-1, -1].legend(
        title="$q$",
        handles=reversed(handles),
        labels=reversed(labels),
        ncols=1,
        fontsize=7,
        labelspacing=0.1,
    )
    [ax.set_yticks([0.25, 0.5, 0.75, 1]) for ax in axxs[:, 0]]
    [
        ax.set_xticks(
            (_dftix["index"] + (bar_width * (n_bars - 1)) / 2)[i % 2 :: 2],
            [f"{p:g}" for p in _dftix["p"]][i % 2 :: 2],
            # _dftix.select(pl.format("{}", "p")).to_series().to_list(),
            # rotation=45,
            # ha="right",
            # rotation_mode="anchor",
        )[0].set_in_layout(False)
        for i, ax in enumerate(axxs[-1])
    ]

    [ax.set_xlabel("$p$") for ax in axxs[-1]]
    return fig
