import inspect
import zipfile
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


# example plotname = "mnist.tsne.summary". The middle part must not
# contain dots as part of the arguments.
def deplist(plotname):
    depdict = deps(plotname)

    srcfiles = depdict.pop("srcfiles")
    return list(depdict.values()) + srcfiles


def deps(plotname):
    dataset, algo, _name = plotname.name.split(".")
    assert _name == "summary"

    path = Path("../runs") / dataset / algo
    depdict = {k: path / k / "1.zip" for k in ["lin", "knn", "recall"]}
    depdict["embedding"] = path / "1.zip"
    depdict["data"] = path.parent / "1.zip"

    depdict["srcfiles"] = [
        inspect.getfile(f) for f in [inspect, Path, np, plt]
    ]
    return depdict


def plot_path(plotname, outfile, format="pdf"):
    files = deps(plotname)

    embedding = np.load(files["embedding"])["embedding"]
    labels = np.load(files["data"])["labels"]
    accd = dict()
    for k in ["lin", "knn", "recall"]:
        with zipfile.ZipFile(files[k]) as zf:
            with zf.open("score.txt") as f:
                acc = float(f.read())
        accd[k] = acc

    return plot(embedding, labels, accd, outfile=outfile, format=format)


def plot(embedding, labels, accd, outfile=None, format="pdf"):
    fig, ax = plt.subplots()

    ax.scatter(*embedding.T, c=labels, alpha=0.6, rasterized=True)

    txt = "\n".join(f"{k} = {v:.1%}" for k, v in accd.items())
    ax.text(
        1,
        1,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        ma="right",
        family="monospace",
    )

    fig.savefig(outfile, format=format)
