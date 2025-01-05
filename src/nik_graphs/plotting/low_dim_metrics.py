from pathlib import Path


def deplist(dispatch: Path):
    assert dispatch.name == "low_dim_metrics"
    return ["../dataframes/all_layouts.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    h5file = deplist(plotname)[0]

    with h5py.File(h5file) as h5:
        return plot(h5, outfile=outfile, format=format)


def plot(h5, outfile, format="pdf"):
    from collections import defaultdict

    import matplotlib as mpl
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import add_letters

    keys = ["recall", "knn", "lin"]

    datadict = defaultdict(list)
    for ds in h5:
        layout_keys = [k for k in h5["cora"] if k not in ["edges", "labels"]]
        n_edges = len(h5[ds]["edges/row"])
        for lk in layout_keys:
            for key in keys:
                acc = h5[ds][lk].attrs[key]
                datadict["dataset"].append(ds)
                datadict["layout_name"].append(lk)
                datadict["n_edges"].append(n_edges)
                datadict["metric"].append(key)
                datadict["score"].append(acc)

    df = pl.DataFrame(datadict)
    legend = ["legend"] * len(keys)
    fig, axd = plt.subplot_mosaic(
        [keys, legend], figsize=(3.25, 1.1), height_ratios=[1, 0.05]
    )

    ax_legend = axd.pop("legend")
    add_letters(axd.values())
    for key, ax in axd.items():
        ax.set(title=key, xscale="log", xlabel="number of edges")
        ax.yaxis.set_major_formatter(
            mpl.ticker.PercentFormatter(1, decimals=0)
        )

        df_ = df.filter(pl.col("metric") == key)
        for (lk,), df_ in df_.group_by("layout_name", maintain_order=True):
            df_ = df_.sort(by="n_edges")
            ax.plot(*df_[["n_edges", "score"]], label=lk, marker="o")

    handles, labels = ax.get_legend_handles_labels()
    ax_legend.set_axis_off()
    ax_legend.legend(
        handles=handles,
        labels=labels,
        ncols=len(handles),
        loc="center",
        borderaxespad=0,
        borderpad=0,
        labelspacing=0,
        # fontsize="small",
    )
    # prop=dict(family="Roboto Condensed"),

    fig.savefig(outfile, format=format)
