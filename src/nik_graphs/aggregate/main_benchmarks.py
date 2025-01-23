def deplist(dispatch=None):
    return ["all_benchmarks.parquet"]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_in = pl.scan_parquet(deplist()[0])
    df = df_in.filter(
        (pl.col("name") == "cne,temp=0.05")
        | ~pl.col("name").str.starts_with("cne"),
    )

    if outfile is not None:
        df.sink_parquet(outfile)

    return df
