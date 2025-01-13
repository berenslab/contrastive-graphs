from pathlib import Path


def deplist(dispatch: Path):
    return ["../dataframes/high_dim_benchmarks.parquet"]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist(plotname)[0])

    return plot(df, outfile=outfile, format=format)


def plot(df_full, outfile, format="pdf"):

    keys = ["knn", "recall", "lin"]

    fig = plot_bars(df_full, keys)
    fig.savefig(outfile, format=format)


def plot_bars(df_full, keys, x_sort_col="n_edges"):
    import matplotlib as mpl
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import add_letters, translate_plotname

    n_bars = len(df_full["name"].unique())
    bar_width = 1 / (n_bars + 1.62)

    fig, axd = plt.subplot_mosaic(
        [keys + ["legend"]],
        figsize=(6.75, 1.5),
        width_ratios=[1] * len(keys) + [0.1],
        constrained_layout=dict(h_pad=0, w_pad=0),
    )
    ax_legend = axd.pop("legend")
    for key, ax in axd.items():
        df_metric = df_full.group_by(
            ["dataset", "name"], maintain_order=True
        ).agg(
            pl.first(x_sort_col),
            pl.mean(key).alias("mean"),
            pl.std(key).alias("std"),
        )
        for i, ((_,), df) in enumerate(
            df_metric.group_by("name", maintain_order=True)
        ):
            df = df.sort(by=x_sort_col)
            x, m, std = df.with_row_index()[["index", "mean", "std"]]
            ax.yaxis.set_major_formatter(
                mpl.ticker.PercentFormatter(1, decimals=0)
            )
            label = translate_plotname(df["name"][0], _return="identity")
            ax.bar(x + i * bar_width, m, label=label, width=bar_width)
            ax.errorbar(
                x + i * bar_width,
                m,
                yerr=std,
                fmt="none",
                ecolor="xkcd:dark grey",
                zorder=5,
            )

        _dftix = (
            df_full[["dataset", x_sort_col]]
            .unique()
            .sort(x_sort_col)
            .with_row_index()[["dataset", "index"]]
        )
        ax.set_xticks(
            _dftix["index"] + (bar_width * (n_bars - 1)) / 2,
            [translate_plotname(d) for d in _dftix["dataset"]],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_title(translate_plotname(key), family="Roboto")
        [ax.axhline(y, color="white") for y in [0.25, 0.5, 0.75]]
        ax.spines.left.set_visible(False)
        ax.tick_params("both", length=0, labelsize=8)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.update_datalim([(0, 1)])
        ax.hlines(
            [0] * len(_dftix),
            xmin=_dftix["index"] - bar_width / 2,
            xmax=_dftix["index"] + bar_width * n_bars - bar_width / 2,
            lw=plt.rcParams["axes.linewidth"],
            color="black",
            clip_on=False,
        )
        ax.spines.bottom.set_visible(False)
        ax.margins(x=0)

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.set_axis_off()
    ax_legend.legend(
        handles=handles,
        labels=labels,
        ncols=1,
        loc="center",
        borderaxespad=0,
        borderpad=0,
        labelspacing=0,
        columnspacing=0.5,
        handletextpad=0.25,
        handlelength=1.25,
        fontsize=7,
    )
    add_letters(axd.values())
    return fig
