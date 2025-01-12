def deplist(dispatch=None):
    return [f"{d}_dim_benchmarks.parquet" for d in ["low", "high"]]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_low, df_high = [pl.read_parquet(f) for f in deplist()]

    df_low1 = df_low.with_columns(
        pl.lit(None, dtype=float).alias("learned_temp"),
        pl.col("name").alias("run_name"),
        pl.lit(2).alias("dim"),
    )
    sortcol = sorted(df_low1.columns)
    df_low2 = df_low1.select(sortcol)

    df_high1 = df_high.with_columns(pl.lit(128).alias("dim"))
    df_high2 = df_high1.select(sortcol)

    df = pl.concat((df_low2, df_high2))

    if outfile is not None:
        df.write_parquet(outfile)

    return df
