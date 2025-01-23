def deplist(dispatch=None):
    return [f"{d}_dim_benchmarks.parquet" for d in ["low", "high"]]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_low, df_high = [pl.scan_parquet(f) for f in deplist()]
    colnames = df_high.collect_schema().names() + ["dim"]

    def sorted_df(df: pl.DataFrame):
        return df

    df_low1 = df_low.with_columns(
        pl.lit(None, dtype=float).alias("learned_temp"),
        pl.lit(2).alias("dim"),
    ).select(colnames)

    df_high1 = df_high.with_columns(pl.lit(128).alias("dim")).select(colnames)

    df = pl.concat((df_low1, df_high1))

    if outfile is not None:
        df.sink_parquet(outfile)

    return df
