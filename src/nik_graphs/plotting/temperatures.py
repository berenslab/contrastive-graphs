import string
from pathlib import Path

import matplotlib as mpl
import polars as pl
from matplotlib import pyplot as plt

from ..plot import letter_dict


# example plotname = "temperatures"
def deplist(plotname):
    # dataset, algo, _name = plotname.name.split(".")
    assert plotname.name == "temperatures"

    path = Path("../dataframes/temp_evals.csv")

    return [path]


def plot_path(plotname, outfile, format="pdf"):
    deps = deplist(plotname)
    df = pl.read_csv(deps[0])

    return plot(df, outfile=outfile, format=format)


def plot(df, outfile, format="pdf"):
    datasets = df["dataset"].unique()
    n_rows = len(datasets)
    keys = ["lin", "knn", "recall", "loss"]
    n_cols = len(keys)
    fig, axxs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5, 1 * n_rows),
        # width_ratios=[1, 1, 1, 1, 0.25],
        squeeze=False,
    )

    cmap = plt.get_cmap("copper")
    norm = mpl.colors.LogNorm(df["temp"].min(), df["temp"].max())

    dfg = df.filter(pl.col("epoch") == df["epoch"].max()).group_by(["dataset"])
    it = enumerate(zip(axxs, dfg))
    letters = iter(string.ascii_lowercase)
    for i, (axs, (grpkey, df_)) in it:
        print(grpkey)

        for ax, key in zip(axs, keys):
            dfgg = df_.group_by("temp")
            temp, mean = dfgg.agg(pl.mean(key)).sort(by="temp")[["temp", key]]
            std = dfgg.agg(pl.std(key)).sort(by="temp")[key]
            ax.scatter(temp, mean, c=cmap(norm(temp)), zorder=5, s=15)
            ax.plot(temp, mean, c="xkcd:grey")
            ax.fill_between(
                temp,
                mean - std,
                mean + std,
                color="xkcd:grey",
                alpha=0.62,
                ec=None,
            )
            letter = next(letters)
            ax.set_title(letter, **letter_dict())
            ax.set(ylabel=key, xlabel="temperature", xscale="log")
            if key != "loss":
                ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))

        # if i == 0:
        #     handles, labels = ax.get_legend_handles_labels()
        #     axs[-1].legend(handles=handles, labels=labels, loc="center")
        # axs[-1].set_axis_off()
    fig.savefig(outfile, format=format)
