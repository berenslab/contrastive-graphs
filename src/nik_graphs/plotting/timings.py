import inspect

from ..plot import add_letters, translate_plotname


def deplist(plotname=None):
    return [
        "../dataframes/main_benchmarks.parquet",
        inspect.getfile(translate_plotname),
    ]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist()[0])

    fig = plot_bars(df)
    fig.savefig(outfile, format=format, metadata=dict(CreationDate=None))


def plot_bars(df_full, x_sort_col="n_edges"):
    import numpy as np
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import name2color, translate_plotname

    df1 = df_full.filter(~pl.col("name").str.starts_with("spectral"))

    panels = ["128", "2"]
    mosaic = np.array([[d, f"legend{d}"] for d in panels]).reshape(1, -1)
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(5.5, 1.3),
        width_ratios=[1, 0.1] * len(panels),
        sharey=True,
        constrained_layout=dict(w_pad=0, h_pad=0),
    )
    for key in panels:
        ax = axd[key]
        df_metric = (
            df1.filter(pl.col("dim") == int(key))
            .group_by(["dataset", "name"], maintain_order=True)
            .agg(
                pl.first(x_sort_col),
                pl.mean("time").alias("mean"),
                pl.std("time").alias("std"),
            )
        )
        n_bars = len(df_metric["name"].unique())
        bar_width = 1 / (n_bars + 1.62)

        for i, ((_,), df) in enumerate(
            df_metric.group_by("name", maintain_order=True)
        ):
            df = df.sort(by=x_sort_col)
            x, m, std = df.with_row_index()[["index", "mean", "std"]]
            label = translate_plotname(df["name"][0], _return="identity")
            color = name2color(df["name"][0])
            kwargs = dict(label=label, width=bar_width, color=color)
            ax.bar(x + i * bar_width, m, **kwargs)
            ax.errorbar(
                x + i * bar_width,
                m,
                yerr=std,
                fmt="none",
                ecolor="xkcd:dark grey",
                zorder=5,
            )

        ax.set_title(f"{key}D", family="Roboto")
        ax.set_yscale("log")
        ax.spines.left.set_visible(False)

        _dftix = (
            df_metric[["dataset", x_sort_col]]
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
        ax.tick_params(
            "both", which="both", length=0, labelsize=8, labelleft=True
        )
        ax.yaxis.set_major_formatter("{x:,g} s")
        ax.hlines(
            [0] * len(_dftix),
            xmin=_dftix["index"] - bar_width / 2,
            xmax=_dftix["index"] + bar_width * n_bars - bar_width / 2,
            lw=plt.rcParams["axes.linewidth"],
            color="black",
            clip_on=False,
            transform=ax.get_xaxis_transform(),
        )
        [ax.axhline(y, color="white") for y in [10**i for i in range(1, 5)]]
        ax.spines.bottom.set_visible(False)
        ax.margins(x=0)
        ax.set_ylim(1, None)

        handles, labels = ax.get_legend_handles_labels()
        axd[f"legend{key}"].set_axis_off()
        axd[f"legend{key}"].legend(
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

    axd["2"].set_ylabel("Runtime", fontsize=8)
    add_letters(axd[k] for k in panels)
    return fig
