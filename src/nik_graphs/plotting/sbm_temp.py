from pathlib import Path

TEMPS = [0.5, 0.05]
METRICS = ["knn", "recall", "ftsne"]


def deplist(plotname=None):
    sbm = Path(
        "../runs/sbm,n_pts=8000,n_blocks=10,p_intra=0.0025,p_inter=5e-6"
    )

    tempstrs = ["" if t == 0.5 else f",temp={t}" for t in TEMPS]
    cnes = ["cne" + t + ",n_epochs=10,detailed=1" for t in tempstrs]
    return [sbm / cne / f"{m},all=1/1.zip" for cne in cnes for m in METRICS]


def plot_path(plotname, outfile, format="pdf"):
    import itertools
    import zipfile

    import numpy as np
    import polars as pl

    deps = deplist(plotname)

    dfs = []
    for fname, (temp, metric) in zip(deps, itertools.product(TEMPS, METRICS)):
        zpath = zipfile.Path(fname)
        if metric == "ftsne":
            continue
        with (zpath / "scores.csv").open() as f:
            df_ = pl.read_csv(f).with_columns(
                temp=pl.lit(temp),
                metric=pl.lit(metric),
            )
        dfs.append(df_)

    df = pl.concat(dfs, how="vertical").pivot("metric", values="score")

    # https://stackoverflow.com/questions/72821244/
    # polars-get-grouped-rows-where-column-value-is-maximum#72821688
    dff = df.filter(pl.col("recall") == pl.max("recall").over("temp"))
    emb_pts = dff["step"]

    pdict = dict()
    for fname, (temp, metric) in zip(deps, itertools.product(TEMPS, METRICS)):
        pdict[fname.parent.parent] = 1

    embd = dict()
    for key in pdict.keys():
        npz = np.load(key / "ftsne,all=1/1.zip")
        embd[key] = {
            step: npz[f"embeddings/step-{step:05d}"] for step in emb_pts
        }
    labels = np.load(key.parent / "1.zip")["labels"]

    fig = plot(df, embd, labels)
    fig.savefig(outfile, format=format)


def plot(df_full, embd, labels):
    import matplotlib as mpl
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import letter_dict, translate_plotname

    mosaic = "cd\nzz\nab"
    fig, axd = plt.subplot_mosaic(
        mosaic, figsize=(3.25, 3), constrained_layout=dict(w_pad=0)
    )

    plot_ax = axd["z"]
    letters = iter("abcdef")
    for i, (((temp,), df), (k, vdict)) in enumerate(
        zip(
            df_full.group_by("temp", maintain_order=True),
            embd.items(),
        )
    ):
        (line,) = plot_ax.plot(*df[["step", "recall"]], label=f"{temp}")
        plot_ax.text(
            df["step"].max(),
            df["recall"][-1],
            f" {temp}",
            clip_on=False,
            ha="left",
            va="center",
        )

        for step, emb in vdict.items():
            letter = next(letters)
            ax = axd[letter]
            ax.scatter(emb[:, 0], emb[:, 1], c=labels, rasterized=True)
            ax.set_aspect(1)
            ax.set_axis_off()

            xy = (step, df.filter(pl.col("step") == step)["recall"][0])
            annot = mpl.text.Annotation(
                "",
                xy,
                (0.5, 1 - i),
                xycoords=plot_ax.transData,
                textcoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->"),
            )
            fig.add_artist(annot)
            plot_ax.scatter(
                [xy[0]],
                [xy[1]],
                c=line.get_color(),
                marker="o",
                s=4,
                clip_on=False,
            )

    plot_ax.set(
        xlabel="step",
        ylabel=translate_plotname("recall"),
        xlim=(0, df["step"].max()),
    )
    plot_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    plot_ax.spines.left.set_bounds(0.4, 0.6)
    ld = letter_dict()
    ld.pop("loc")
    [
        ax.text(0, 1, ltr, transform=ax.transAxes, ha="left", va="top", **ld)
        for ltr, ax in zip("abcdefg", axd.values())
    ]
    return fig
