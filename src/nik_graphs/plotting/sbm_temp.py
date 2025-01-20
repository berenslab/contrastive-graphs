from pathlib import Path


def deplist(plotname=None):
    return ["../dataframes/sbm_temp.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    with h5py.File(deplist(plotname)[0]) as h5:
        fig = plot(h5)

    fig.savefig(outfile, format=format)


def plot(h5):
    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt

    from ..plot import letter_dict, translate_plotname

    mosaic = "ab\nzz\nde"
    letters = iter("abde")
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=(3.25, 3), constrained_layout=dict(w_pad=0)
    )

    plot_ax = axd["z"]

    labels = h5["labels"]
    for i, (temp_str, h5_temp) in enumerate(reversed(h5.items())):
        if temp_str == "labels":
            continue

        steps = np.asanyarray(h5_temp["step"])
        recalls = np.asanyarray(h5_temp["recall"])
        (line,) = plot_ax.plot(steps, recalls)
        plot_ax.text(
            steps.max(),
            recalls[-1],
            f" $τ={{}}${temp_str}",
            clip_on=False,
            ha="left",
            va="center",
        )

        for key, emb in h5_temp.items():
            if not key.startswith("step-"):
                continue
            n = len("step-")
            step = int(key[n:])
            recall = recalls[steps == step]

            letter = next(letters)
            ax = axd[letter]
            ax.scatter(emb[:, 0], emb[:, 1], c=labels, rasterized=True)
            ax.set_aspect(1)
            ax.set_axis_off()
            ax.margins(0)

            x, y = step, recall
            plot_ax.scatter(
                [x],
                [y],
                c=line.get_color(),
                marker="o",
                s=4,
                clip_on=False,
            )
            txtkwargs = dict(
                va="bottom" if letter in ["d", "b"] else "top",
                ha="right" if letter == "d" else "center",
            )
            plot_ax.text(x, y, letter, **txtkwargs)

    plot_ax.set(
        xlabel="step",
        ylabel=translate_plotname("recall"),
        xlim=(0, steps.max()),
    )
    plot_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    plot_ax.spines.left.set_bounds(0, 0.75)
    ld = letter_dict()
    ld.pop("loc")
    [
        ax.text(0, 1, ltr, transform=ax.transAxes, ha="left", va="top", **ld)
        for ltr, ax in zip("abcdefg", axd.values())
    ]
    return fig
