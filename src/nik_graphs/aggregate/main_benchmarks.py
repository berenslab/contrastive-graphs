def deplist(dispatch=None):
    return ["all_benchmarks.parquet"]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_in = pl.scan_parquet(deplist()[0])
    df = df_in.filter(
        ~pl.col("run_name").is_in(["cne", "cne,loss=infonce-temp"])
    ).select(
        pl.all().exclude("name"),
        pl.col("name").replace("CNE, τ=0.05", "graph CNE"),
    )

    if outfile is not None:
        df.sink_parquet(outfile)

    return df
