import inspect
from pathlib import Path

import matplotlib as mpl
import polars as pl
from matplotlib import pyplot as plt


# example plotname = "temperatures"
def deplist(plotname):
    # dataset, algo, _name = plotname.name.split(".")
    assert plotname.name == "temperatures"

    path = Path("../dataframes/temp_evals.csv")

    return [path] + [inspect.getfile(f) for f in [inspect, Path, plt]]


def plot_path(plotname, outfile, format="pdf"):
    deps = deplist(plotname)
    df = pl.read_csv(deps[0])

    return plot(df, outfile=outfile, format=format)


def plot(df, outfile=None, format="pdf"):
    keys = ["lin", "knn", "recall"]
    fig, axd = plt.subplot_mosaic(
        [keys + ["legend"]], figsize=(5.5, 1.25), width_ratios=[1, 1, 1, 0.25]
    )

    cmap = plt.get_cmap("copper")
    norm = mpl.colors.LogNorm(df["temp"].min(), df["temp"].max())
    for (temp,), df_ in df.group_by("temp", maintain_order=True):
        df__ = df_.group_by("epoch")
        for key in keys:
            color = cmap(norm(temp))
            m = df__.agg(pl.mean(key)).sort(by="epoch")
            std = df__.agg(pl.std(key)).sort(by="epoch")[key]
            ax = axd[key]
            ax.plot(*m, c=color, label=f"{temp:g}")
            ax.fill_between(
                m["epoch"],
                m[key] - std,
                m[key] + std,
                color=color,
                ec=None,
                alpha=0.62,
            )
    for key in keys:
        ax = axd[key]
        ax.set_ylabel(key)
        ax.set_xlabel("epochs")
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    handles, labels = axd["knn"].get_legend_handles_labels()
    axd["legend"].legend(handles=handles, labels=labels, loc="center")
    axd["legend"].set_axis_off()
    fig.savefig(outfile, format=format)
