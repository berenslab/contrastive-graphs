def deplist(plotname=None):
    return ["../dataframes/node2vec_pq.parquet"]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist(plotname)[0])

    fig = plot(df)
    fig.savefig(outfile, format=format)


def plot(df_full):
    import matplotlib as mpl
    import numpy as np
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import translate_plotname  # translate_acc_short,

    metrics = ["recall", "knn", "lin"]
    # norms = [
    #     mpl.colors.LogNorm(df_full[m].min(), df_full[m].max()) for m in metrics
    # ]

    fig = plt.figure(figsize=(6.75, 2.5))
    figs = fig.subfigures(3, 3)

    for ((dataset,), df), sfig in zip(
        df_full.group_by("dataset", maintain_order=True), figs.flat
    ):
        axs = sfig.subplots(1, 3)
        sfig.suptitle(translate_plotname(dataset))

        n_bars = len(df["q"].unique())
        bar_width = 1 / (n_bars + 1.62)
        _dftix = df.unique("p", maintain_order=True).with_row_index()
        cmap = plt.get_cmap("copper")
        colors = [cmap(x) for x in np.linspace(0.3, 1, num=n_bars)]

        for m, ax in zip(metrics, axs):
            # ax.set_title(m)
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
                    label=f"${q=}$",
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
            ax.set_yticks([])
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            ax.hlines(
                [0] * len(_dftix),
                xmin=_dftix["index"] - bar_width / 2,
                xmax=_dftix["index"] + bar_width * n_bars - bar_width / 2,
                lw=plt.rcParams["axes.linewidth"],
                color="black",
                clip_on=False,
            )
            ax.set_xticks(
                _dftix["index"] + (bar_width * (n_bars - 1)) / 2,
                [f"{p:g}" for p in _dftix["p"]],
                # _dftix.select(pl.format("{}", "p")).to_series().to_list(),
                # rotation=45,
                # ha="right",
                # rotation_mode="anchor",
            )

            [ax.spines[x].set_visible(False) for x in ["bottom", "left"]]
            ax.margins(x=0)

    handles, labels = ax.get_legend_handles_labels()
    figs[-1, -1].legend(handles=handles, labels=labels, ncols=2)
    [fig.get_axes()[0].set_yticks([0.25, 0.5, 0.75, 1]) for fig in figs[:, 0]]
    # [ax.set_xlabel("p") for fig in figs[-1] for ax in fig.get_axes()]
    # ax0 = fig.get_axes()[0]
    # [ax.sharey(ax0) for ax in fig.get_axes()[1:]]
    return fig
