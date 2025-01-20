DATASETS = ["photo", "computer"]
ROWNORM = ["", ",row_norm=0"]


# example plotname = "mnist.tsne.summary". The middle part must not
# contain dots as part of the arguments.
def deplist(plotname):
    return [dep for ddict in deps(plotname).values() for dep in ddict.values()]


def deps(plotname):
    import itertools
    from pathlib import Path

    assert plotname.name == "tsne_rownorm"

    ddict = dict()
    for dataset, rn in itertools.product(DATASETS, ROWNORM):
        path = Path("../runs") / dataset / ("tsne" + rn)
        depdict = {k: path / k / "1.zip" for k in ["lin", "knn", "recall"]}
        depdict["embedding"] = path / "1.zip"
        depdict["data"] = path.parent / "1.zip"
        ddict[dataset, rn] = depdict
    return ddict


def plot_path(plotname, outfile, format="pdf"):
    return plot(deps(plotname), outfile=outfile, format=format)


def plot(depd, outfile=None, format="pdf"):
    import itertools
    import zipfile

    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import linalg, sparse

    from ..plot import add_letters, translate_acc_short, translate_plotname

    fig, axs = plt.subplots(len(DATASETS), len(ROWNORM), figsize=(3.5, 3.75))
    prev_dataset = None
    for (dataset, rn), ax in zip(
        itertools.product(DATASETS, ROWNORM), axs.flat
    ):
        norm_title = "(default)" if rn == "" else "whole-matrix norm."
        ds_title = translate_plotname(dataset)
        tsne_title = translate_plotname("tsne")
        ax.set_title(f"{ds_title}, {tsne_title}\n{norm_title}")
        anchor = np.load(depd[dataset, ""]["embedding"])["embedding"]

        depdict = depd[dataset, rn]
        embedding = np.load(depdict["embedding"])["embedding"]

        if dataset != prev_dataset:
            A = sparse.load_npz(depdict["data"]).tocoo()
            npz = np.load(depdict["data"])
            labels = npz["labels"]
            prev_dataset = dataset

        rot, _scale = linalg.orthogonal_procrustes(embedding, anchor)
        embedding = embedding @ rot.round(10)

        ax.scatter(*embedding.T, c=labels, rasterized=True)
        ax.set_axis_off()
        ax.axis("equal")

        def key2acc(k):
            zpath = zipfile.Path(depdict[k]) / "score.txt"
            return float(zpath.read_text())

        txt = "\n".join(
            f"{translate_acc_short(k)}$ = ${key2acc(k):.1%}"
            for k in ["knn", "recall"]
        )
        ax.text(
            1,
            1,
            txt,
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="top",
            ma="right",
        )

        pts = np.hstack((embedding[A.row], embedding[A.col])).reshape(
            len(A.row), 2, 2
        )
        lines = mpl.collections.LineCollection(
            pts,
            alpha=0.05,
            color="xkcd:dark grey",
            antialiaseds=True,
            zorder=0.9,
            rasterized=True,
        )
        ax.add_collection(lines)

    add_letters(axs.flat)
    fig.savefig(outfile, format=format)
