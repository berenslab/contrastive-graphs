from pathlib import Path


def deplist(dispatch: Path):
    return ["../dataframes/all_layouts.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py
    from matplotlib import pyplot as plt

    h5file = deplist(plotname)[0]

    with h5py.File(h5file) as h5:
        with plt.rc_context({"font.family": ["Roboto", "Noto Sans"]}):
            return plot(h5, outfile=outfile, format=format)


def plot(h5, outfile, datasets=["computer", "photo"], format="pdf"):
    import itertools

    import matplotlib as mpl
    import matplotlib.patheffects
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import linalg

    from ..plot import (
        add_scalebars,
        letter_dict,
        letter_iterator,
        translate_acc_short,
        translate_plotname,
    )

    letters = letter_iterator()
    ldict = letter_dict()
    ldict.pop("horizontalalignment", None)
    keys = ["tsne", "drgraph", "fa2", "tfdp"]
    fig, axs = plt.subplots(
        len(datasets),
        len(keys),
        figsize=(6, 3),
        constrained_layout=dict(w_pad=0, h_pad=0),
    )

    for (dataset, key), ax in zip(itertools.product(datasets, keys), axs.flat):
        h5_ds = h5[dataset]
        labels = h5_ds["labels"]
        row = h5_ds["edges/row"]
        col = h5_ds["edges/col"]

        anchor = h5_ds["tsne"]  # take any array

        if dataset == datasets[0]:
            ax.set_title(translate_plotname(key))
        if key == keys[0]:
            ax.set_ylabel(
                translate_plotname(dataset),
                fontsize=plt.rcParams["axes.titlesize"],
            )

        data = np.array(h5_ds[key])
        rot, _scale = linalg.orthogonal_procrustes(data, anchor)
        data = data @ rot.round(10)
        ax.scatter(*data.T, c=labels, rasterized=True)
        ax.set_title(next(letters), **ldict, ha="left")
        ax.axis("equal")
        add_scalebars(ax, hidex=False, hidey=False)
        [ax.spines[x].set_visible(False) for x in ["left", "bottom"]]
        ax.tick_params(
            "both",
            which="both",
            length=0,
            labelleft=False,
            labelbottom=False,
        )

        lines = (
            f"{translate_acc_short(k)}$ = ${v:5.1%}"
            for k, v in h5_ds[key].attrs.items()
            if k != "lin"
        )
        txt = "\n".join(sorted(lines, key=len, reverse=True))
        p_eff = [mpl.patheffects.withStroke(linewidth=1.1, foreground="white")]

        ax.text(
            1,
            1,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            path_effects=p_eff,
        )

        pts = np.hstack((data[row], data[col])).reshape(len(row), 2, 2)
        lines = mpl.collections.LineCollection(
            pts,
            alpha=0.05,
            color="xkcd:dark grey",
            antialiaseds=True,
            zorder=0.9,
            rasterized=True,
        )
        ax.add_collection(lines)

    fig.savefig(outfile, format=format)
