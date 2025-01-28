def deplist(plotname=None):
    return ["../dataframes/node2vec_pq.parquet"]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist(plotname)[0])

    fig = plot(df)
    fig.savefig(outfile, format=format)


def plot(df_full):
    import matplotlib as mpl
    import numpy as np
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import translate_plotname  # translate_acc_short,

    metrics = ["recall", "knn", "lin"]
    # norms = [
    #     mpl.colors.LogNorm(df_full[m].min(), df_full[m].max()) for m in metrics
    # ]

    fig = plt.figure(figsize=(6.75, 2.5))
    figs = fig.subfigures(3, 3)

    for ((dataset,), df), sfig in zip(
        df_full.group_by("dataset", maintain_order=True), figs.flat
    ):
        axs = sfig.subplots(1, 3)
        sfig.suptitle(translate_plotname(dataset))
        for m, ax in zip(metrics, axs):
            # ax.set_title(m)
            df_ = df.pivot("q", index="p", values=m, aggregate_function="mean")
            ps = df_["p"]
            qs = [float(q) for q in df_.select(pl.all().exclude("p")).columns]

            mat = df_.select(pl.all().exclude("p")).to_numpy()

            pix, qix = np.unravel_index(mat.argmax(), mat.shape)

            mappbl = ax.imshow(mat.T, origin="lower", cmap="viridis")
            sfig.colorbar(mappbl, ax=ax)
            ax.tick_params("both", length=0)
            ax.set_xticks([pix], [f"{ps[pix.item()]}"])
            ax.set_yticks([qix], [f"{qs[qix.item()]}"])
            # ax.pcolormesh(ps, qs, mat, cmap="viridis")
            # ax.set_xscale("log", base=2)
            # ax.set_yscale("log", base=2)
            # ax.set_xlim(0.1875, 6)
            # ax.set_ylim(0.1875, 6)
            # ax.set_xticks(ps)
            # ax.set_yticks(ps)
            # # ax.margins(0)
            # ax.set_aspect(1)

    return fig
