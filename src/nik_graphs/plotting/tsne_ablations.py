from pathlib import Path


def deplist(dispatch: Path):
    return ["../dataframes/tsne_ablations.h5", "../dataframes/random_inits.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    deps = deplist(plotname)
    h5file = deps[0]

    with h5py.File(h5file) as h5:
        return plot(h5, outfile=outfile, format=format)


def plot(h5, outfile, *, datasets=["computer", "photo"], format="pdf"):

    import matplotlib as mpl
    import matplotlib.patheffects
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import linalg

    from ..plot import letter_dict, translate_acc_short, translate_plotname

    tr_variants = dict(
        default="Default",
        random_init="Random init.",
        no_row_norm="Whole-matrix norm.",
    )

    fig, axxs = plt.subplots(
        len(datasets),
        len(tr_variants),
        figsize=(4.75, 3.5),
        constrained_layout=dict(h_pad=0, w_pad=0),
    )
    ltrdict = letter_dict()
    ltrdict.update(horizontalalignment="left")
    letters = iter("abcdefgh")

    for (dataset, h5_ds), axs in zip(h5.items(), axxs):
        labels = h5_ds["labels"]
        row = h5_ds["edges/row"]
        col = h5_ds["edges/col"]

        anchor = h5_ds["tsne/default"]  # take any array
        axs[0].set_ylabel(
            translate_plotname(dataset),
            fontsize=plt.rcParams["axes.titlesize"],
        )

        for ax, (var_key, txt) in zip(axs, tr_variants.items()):
            ax.set_title(txt)

            data = np.array(h5[dataset][f"tsne/{var_key}"])
            rot, _scale = linalg.orthogonal_procrustes(data, anchor)
            data = data @ rot.round(10)
            ax.scatter(*data.T, c=labels, rasterized=True)
            ax.axis("equal")
            [ax.spines[x].set_visible(False) for x in ["left", "bottom"]]
            ax.tick_params(
                "both",
                which="both",
                length=0,
                labelleft=False,
                labelbottom=False,
            )

            lines = (
                f"{translate_acc_short(k)}$ = ${v:5.1%}".strip()
                for k, v in h5[dataset][f"tsne/{var_key}"].attrs.items()
                if k != "lin"
            )
            txt = "\n".join(sorted(lines, key=len, reverse=True))
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
            ax.set_title(next(letters), **ltrdict)

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
