from pathlib import Path


def deplist(dispatch: Path):
    assert dispatch.name == "random_inits"
    return ["../dataframes/all_layouts.h5", "../dataframes/random_inits.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    deps = deplist(plotname)
    h5afile = deps[0]
    h5bfile = deps[1]

    with h5py.File(h5afile) as h5a, h5py.File(h5bfile) as h5b:
        return plot(h5a, h5b, outfile=outfile, format=format)


def plot(h5a, h5b, outfile, *, datasets=["computer", "photo"], format="pdf"):
    import itertools

    import matplotlib as mpl
    import matplotlib.patheffects
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import linalg

    from ..plot import letter_dict, translate_acc_short, translate_plotname

    fig, axxs = plt.subplots(
        len(datasets),
        2,
        figsize=(3.5, 3.75),
        constrained_layout=dict(h_pad=0, w_pad=0),
    )
    ltrdict = letter_dict()
    ltrdict.update(horizontalalignment="left")

    inits = dict(default=h5a, random=h5b)
    for (dataset, (initstr, h5)), ltr, ax in zip(
        itertools.product(datasets, inits.items()), "abcdef", axxs.flat
    ):
        labels = h5a[dataset]["labels"]
        row = h5a[dataset]["edges/row"]
        col = h5a[dataset]["edges/col"]

        anchor = h5a[dataset]["tsne"]  # take any array
        # this is the same order as in low_dim_metrics.py

        tstr = f"{translate_plotname(dataset)}, {initstr} init."
        ax.set_title(tstr)

        data = np.array(h5[dataset]["tsne"])
        rot, _scale = linalg.orthogonal_procrustes(data, anchor)
        data = data @ rot.round(10)
        ax.scatter(*data.T, c=labels, rasterized=True)
        ax.axis("equal")
        ax.set_axis_off()

        lines = (
            f"{translate_acc_short(k)}$ = ${v:5.1%}".strip()
            for k, v in h5[dataset]["tsne"].attrs.items()
            if k != "lin"
        )
        txt = "\n".join(sorted(lines, key=len, reverse=True))
        # import sys

        # print(lines, file=sys.stderr)
        # print(txt, file=sys.stderr)
        # p_eff = [mpl.patheffects.withStroke(linewidth=1.1, foreground="white")]
        ax.text(
            1,
            1,
            txt,
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="top",
            ma="right",
            # path_effects=p_eff,
        )
        ax.set_title(ltr, **ltrdict)

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

    fig.savefig(outfile, format=format, metadata=dict(CreationDate=None))
