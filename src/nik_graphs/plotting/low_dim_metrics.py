from pathlib import Path


def deplist(dispatch: Path):
    import inspect

    from .high_dim_metrics import plot_bars

    assert dispatch.name == "low_dim_metrics"
    return [
        "../dataframes/low_dim_benchmarks.parquet",
        inspect.getfile(plot_bars),
    ]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist(plotname)[0])

    return plot(df, outfile=outfile, format=format)


def plot(df, outfile, format="pdf"):

    import polars as pl

    from .high_dim_metrics import plot_bars

    keys = ["knn", "recall", "lin"]
    order = "tsne sgtsnepi drgraph fa2 tfdp spectral".split()
    df_order = pl.DataFrame(dict(name=order)).with_row_index()

    df_ = df.join(df_order, on="name").sort("index").drop("index")

    fig = plot_bars(df_, keys)
    fig.savefig(outfile, format=format)
