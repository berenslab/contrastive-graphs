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
    depvals = deps(plotname).values()
    return [f for lis in depvals for f in lis]


def deps(plotname):
    # dataset, algo, _name = plotname.name.split(".")
    assert plotname.name == "temperatures"

    dataset = "mnist"

    from ..modules.cne import tsimcne_nonparam

    sig = inspect.signature(tsimcne_nonparam)
    default_temp = sig.parameters["temp"].default
    default_n_epochs = sig.parameters["n_epochs"].default

    # rng = np.random.default_rng(333**3)
    # rints = rng.integers(10000, size=2)
    rstrs = [""]  # + [f",random_state={r}" for r in rints]

    n_epochs = f",n_epochs={N_EPOCHS}" if default_n_epochs != N_EPOCHS else ""
    path = Path("../runs") / dataset
    paths = []
    for temp, rstr in itertools.product(_TEMPERATURES, rstrs):
        tempstr = f",temp={temp}" if temp != default_temp else ""
        paths.append(path / ("cne,metric=cosine" + n_epochs + tempstr + rstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths] for k in ["lin", "knn", "recall"]
    }

    depdict["srcfiles"] = [
        inspect.getfile(f) for f in [inspect, Path, np, plt, zipfile]
    ]
    return depdict


def plot_path(plotname, outfile, format="pdf"):
    files = deps(plotname)
    deps.pop("srcfiles")

    # labels = np.load(files["data"])["labels"]
    df_dict = dict()
    for k, v in deps.items():
        df_ = pl.DataFrame(dict(temp=[], epoch=[], k=[]))
        for temp, zipf in zip(_TEMPERATURES, v):
            with zipfile.ZipFile(zipf) as zf:
                with zf.open("scores.csv") as f:
                    df1 = pl.read_csv(f).drop("step")

                with zf.open("score.txt") as f:
                    score = float(f.read())
                    df2 = pl.DataFrame(dict(epoch=[N_EPOCHS], score=[score]))
            df__ = df1.vstack(df2).with_columns(pl.lit(temp).alias("temp"))
            df_ = df_.vstack(df__.rename(dict(score=k)))
        df_dict[k] = df_

    df = pl.concat(
        [df_dict[k][["temp", "epoch"]]] + [df[k] for k, df in df_dict.items()],
        how="horizontal",
    )

    return plot(df, outfile=outfile, format=format)


def plot(df, outfile=None, format="pdf"):
    keys = ["lin", "knn", "recall"]
    fig, axd = plt.subplot_mosaic(keys, figsize=(5.5, 2))

    # TODO: well, I need to actually plot here

    fig.savefig(outfile, format=format)
