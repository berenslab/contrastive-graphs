import inspect
from pathlib import Path

import matplotlib as mpl
import polars as pl
from matplotlib import pyplot as plt

_TEMPERATURES = [x * 10**i for i in range(-4, 1) for x in [1, 5]]
N_EPOCHS = 30


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
    fig, axd = plt.subplot_mosaic([keys], figsize=(5.5, 2))

    cmap = plt.get_cmap("magma")
    norm = mpl.colors.LogNorm(df["temp"].min(), df["temp"].max())
    for temp, df_ in df.group_by("temp", maintain_order=True):
        for key in keys:
            ax = axd[key]
            ax.plot(
                df["epoch"], df[key], color=cmap(norm(temp)), label=f"{temp}"
            )
    for key in keys:
        ax = axd[key]
        ax.set_ylabel(key)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

    fig.savefig(outfile, format=format)
