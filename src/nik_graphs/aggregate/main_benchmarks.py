def deplist(dispatch=None):
    return ["all_benchmarks.parquet"]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_in = pl.scan_parquet(deplist()[0])
    df = df_in.filter(~pl.col("name").is_in(["cne", "cne,loss=infonce-temp"]))

    if outfile is not None:
        df.sink_parquet(outfile)

    return df
