import inspect
import itertools
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

_TEMPERATURES = [x * 10**i for i in range(-4, 1) for x in [1, 5]]
N_EPOCHS = 30


# example plotname = "temperatures"
def deplist(plotname):
    # dataset, algo, _name = plotname.name.split(".")
    assert plotname.name == "temperatures"

    path = Path("../dataframes/temp_evals.csv")

    return [path] + [
        inspect.getfile(f) for f in [inspect, Path, np, plt, zipfile]
    ]


def plot_path(plotname, outfile, format="pdf"):
    deps = deplist(plotname)
    df = pl.read_csv(deps[0])

    return plot(df, outfile=outfile, format=format)


def plot(df, outfile=None, format="pdf"):
    keys = ["lin", "knn", "recall"]
    fig, axd = plt.subplot_mosaic(keys, figsize=(5.5, 2))

    # TODO: well, I need to actually plot here
    df.group_by("temp")

    fig.savefig(outfile, format=format)
