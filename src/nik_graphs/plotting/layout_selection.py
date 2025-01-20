from pathlib import Path


def deplist(dispatch: Path):
    return ["../dataframes/all_layouts.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py
    from matplotlib import pyplot as plt

    dataset, _layout_selection = plotname.name.split(".")
    assert _layout_selection == "layout_selection"

    h5file = deplist(plotname)[0]

    with h5py.File(h5file) as h5:
        with plt.rc_context({"font.family": ["Roboto", "Noto Sans"]}):
            return plot(h5, dataset, outfile=outfile, format=format)


def plot(h5, dataset, outfile, format="pdf"):
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
    fig, axd = plt.subplot_mosaic(
        np.array(keys).reshape(2, 2),
        figsize=(3.25, 3),
        constrained_layout=dict(w_pad=0, h_pad=0),
    )

    h5_ds = h5[dataset]
    labels = h5_ds["labels"]
    row = h5_ds["edges/row"]
    col = h5_ds["edges/col"]
    for key, ax in axd.items():
        anchor = h5_ds["tsne"]  # take any array

        ax.set_title(translate_plotname(key))

        data = np.array(h5_ds[key])
        rot, _scale = linalg.orthogonal_procrustes(data, anchor)
        data = data @ rot.round(10)
        ax.scatter(*data.T, c=labels, rasterized=True)
        ax.set_title(next(letters), **ldict, ha="left")
        ax.axis("equal")
        add_scalebars(ax)

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
