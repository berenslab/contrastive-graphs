def deplist(plotname=None):
    return ["../dataframes/sbm_temp.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    with h5py.File(deplist(plotname)[0]) as h5:
        fig = plot(h5)

    fig.savefig(outfile, format=format, metadata=dict(Creationdate=None))


def plot(h5):
    import matplotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt

    from ..plot import letter_dict, translate_plotname

    rng = np.random.default_rng(23890147)
    mosaic = "ab\nzz\nde"
    letters = iter("abde")
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(3.25, 2.75),
        constrained_layout=dict(w_pad=0, h_pad=0.005),
    )
    plot_ax = axd["z"]

    labels = np.asanyarray(h5["labels"])
    for i, (temp_str, h5_temp) in enumerate(h5.items()):
        if temp_str in ["edges", "labels"]:
            continue

        batches_per_epoch = h5_temp.attrs["batches_per_epoch"]
        steps = np.asanyarray(h5_temp["step"]) / batches_per_epoch
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

        sample_keys = [key for key in h5_temp if key.startswith("step-")]
        shuf = rng.permutation(len(h5_temp[sample_keys[0]]))
        for key, emb in h5_temp.items():
            if not key.startswith("step-"):
                continue
            n = len("step-")
            step = int(key[n:]) / batches_per_epoch
            recall = recalls[steps == step]

            letter = next(letters)
            ax = axd[letter]
            Y = np.asanyarray(emb)
            ax.scatter(
                *Y[shuf].T,
                c=labels[shuf],
                alpha=0.7,
                rasterized=True,
                clip_on=False,
            )
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
                va="baseline" if letter in ["a", "e"] else "top",
                ha=(
                    "right"
                    if letter == "a" or step == steps.max()
                    else "center"
                ),
            )
            dy = -1 if txtkwargs["va"] == "top" else 1
            t = mpl.transforms.offset_copy(
                plot_ax.transData, fig, 0, dy * 1.75, units="points"
            )
            plot_ax.text(x, y, letter, transform=t, **txtkwargs)

    plot_ax.set(
        ylabel=translate_plotname("recall"),
        xlim=(0 - 0.125, steps.max()),
    )
    plot_ax.spines.bottom.set_bounds(0, steps.max())
    plot_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    plot_ax.yaxis.set_major_locator(
        mpl.ticker.FixedLocator([0, 0.25, 0.5, 0.75])
    )
    plot_ax.set_xlabel("epoch", labelpad=-1.5)
    plot_ax.set_xticks([0, 10])
    plot_ax.set_xticks(range(1, 10), minor=True)
    plot_ax.spines.left.set_bounds(0, 0.75)
    ld = letter_dict()
    ld.pop("loc")
    for ltr, ax in zip("abcdefg", axd.values()):
        if ltr != "c":
            ax.text(
                0, 1, ltr, transform=ax.transAxes, ha="left", va="top", **ld
            )
        else:
            ax.set_title(ltr, **ld, loc="left", pad=0)
    return fig
