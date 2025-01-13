from pathlib import Path


def deplist(dispatch: Path):
    import inspect

    from .high_dim_metrics import plot_bars

    assert dispatch.name == "low_dim_metrics"
    return ["../dataframes/all_layouts.h5", inspect.getfile(plot_bars)]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    h5file = deplist(plotname)[0]

    with h5py.File(h5file) as h5:
        return plot(h5, outfile=outfile, format=format)


def plot(h5, outfile, format="pdf"):
    from collections import defaultdict

    import polars as pl

    from .high_dim_metrics import plot_bars

    keys = ["knn", "recall", "lin"]
    order = "tsne sgtsnepi drgraph fa2 tfdp spectral".split()
    order_dict = {k: i for i, k in enumerate(order)}

    datadict = defaultdict(list)
    for ds in h5:
        layout_keys = [
            k for k in h5["cora"] if k not in ["cne", "edges", "labels"]
        ]
        n_edges = len(h5[ds]["edges/row"])
        for lk in layout_keys:
            datadict["dataset"].append(ds)
            datadict["name"].append(lk)
            datadict["n_edges"].append(n_edges)
            datadict["bar_order"].append(order_dict[lk])
            for key in keys:
                acc = h5[ds][lk].attrs[key]
                datadict[key].append(acc)

    df = pl.DataFrame(datadict).sort("bar_order")

    fig = plot_bars(df, keys)
    fig.savefig(outfile, format=format)
