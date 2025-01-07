from pathlib import Path


def deplist(dispatch: Path):
    return ["../dataframes/all_layouts.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    h5file = deplist(plotname)[0]

    with h5py.File(h5file) as h5:
        return plot(h5, outfile=outfile, format=format)


def plot(h5, outfile, format="pdf"):
    import itertools

    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import linalg

    from ..plot import add_letters, add_scalebars

    fig, axs = plt.subplots(1, 4, figsize=(6.75, 2))
    add_letters(axs.flat)
    datasets = ["computer", "photo"]
    keys = ["tsne", "cne"]

    for (dataset, key), ax in zip(itertools.product(datasets, keys), axs.flat):
        h5_ds = h5[dataset]
        labels = h5_ds["labels"]
        row = h5_ds["edges/row"]
        col = h5_ds["edges/col"]

        anchor = h5_ds["tsne"]  # take any array
        ax.set_title(f"{dataset} {key}")

        data = np.array(h5_ds[key])
        rot, _scale = linalg.orthogonal_procrustes(data, anchor)
        data = data @ rot.round(10)
        ax.scatter(*data.T, c=labels, rasterized=True)
        ax.axis("equal")
        add_scalebars(ax)

        lines = (
            f"{k} = {v:5.1%}"
            for k, v in h5_ds[key].attrs.items()
            if k != "lin"
        )
        txt = "\n".join(sorted(lines, key=len, reverse=True))
        ax.text(
            1,
            1,
            txt,
            transform=ax.transAxes,
            fontsize="x-small",
            family="monospace",
            ha="right",
            va="top",
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
