from pathlib import Path


def deplist(dispatch: Path):
    return ["../dataframes/tsne_cne_evals.parquet"]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist(plotname)[0])

    return plot(df, outfile=outfile, format=format)


def plot(df_full, outfile, format="pdf"):
    import matplotlib as mpl
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import add_letters

    keys = ["knn", "lin", "recall"]  # , "time"]
    legend = ["legend"] * len(keys)

    fig, axd = plt.subplot_mosaic(
        [keys, legend], figsize=(3.25, 1.5), height_ratios=[1, 0.05]
    )
    ax_legend = axd.pop("legend")
    for key, ax in axd.items():
        df_metric = df_full.group_by(
            ["dataset", "run_name"], maintain_order=True
        ).agg(
            pl.first("name", "n_edges"),
            pl.mean(key).alias("mean"),
            pl.std(key).alias("std"),
        )
        for (_,), df in df_metric.group_by("run_name", maintain_order=True):
            df = df.sort(by="n_edges")
            label = df["name"].head(1).item()
            x, m, std = df[["n_edges", "mean", "std"]]
            ax.yaxis.set_major_formatter(
                mpl.ticker.PercentFormatter(1, decimals=0)
            )
            (line,) = ax.plot(x, m, label=label, marker="o")
            ax.fill_between(
                x, m + std, m - std, color=line.get_color(), alpha=0.618
            )
        ax.set(title=key, xscale="log")
        ax.set_title(key, family="Roboto")

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.set_axis_off()
    ax_legend.legend(
        handles=handles,
        labels=labels,
        ncols=3,
        loc="center right",
        borderaxespad=0,
        borderpad=0,
        labelspacing=0,
        columnspacing=0.5,
        handletextpad=0.25,
        handlelength=1.25,
        fontsize=7,
    )
    [ax.set_xlabel("number of edges") for ax in axd.values()]
    add_letters(axd.values())
    fig.savefig(outfile, format=format)
