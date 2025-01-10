from pathlib import Path


def deplist(dispatch: Path):
    assert dispatch.name == "layouts"
    return ["../dataframes/all_layouts.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    h5file = deplist(plotname)[0]

    with h5py.File(h5file) as h5:
        return plot(h5, outfile=outfile, format=format)


def plot(h5, outfile, format="pdf"):
    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import linalg

    from ..plot import translate_plotname

    fig = plt.figure(figsize=(6.75, 1.1 * len(h5)))
    figs = fig.subfigures(len(h5))
    for i, (sfig, dataset) in enumerate(zip(figs, h5)):
        sfig.supylabel(
            translate_plotname(dataset),
            fontsize=plt.rcParams["axes.titlesize"],
        )
        h5_ds = h5[dataset]
        keys = [k for k in h5_ds if k not in ["edges", "labels"]]
        labels = h5_ds["labels"]
        row = h5_ds["edges/row"]
        col = h5_ds["edges/col"]

        axd = sfig.subplot_mosaic([keys])
        anchor = h5_ds["tsne"]  # take any array
        for key, ax in axd.items():
            ax.set_title(translate_plotname(key)) if i == 0 else None

            data = np.array(h5_ds[key])
            rot, _scale = linalg.orthogonal_procrustes(data, anchor)
            data = data @ rot.round(10)
            ax.scatter(*data.T, c=labels, rasterized=True)
            ax.axis("equal")
            ax.set_axis_off()

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
