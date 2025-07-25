def deplist(dispatch=None):
    return [f"{d}_dim_benchmarks.parquet" for d in ["low", "high"]] + [
    ]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_low, df_high = [pl.scan_parquet(f) for f in deplist()]
    colnames = df_high.collect_schema().names() + ["dim", "p", "q", "index"]

    plnone = pl.lit(None, dtype=float)

    df_low1 = (
        df_low.with_row_index()
        .with_columns(
            pl.lit(2).alias("dim"), learned_temp=plnone, p=plnone, q=plnone
        )
        .select(colnames)
    )

    dfix = df_high.with_row_index(
        offset=df_low1.select(pl.len()).collect().item()
    )
    df_high1 = (
        dfix.filter(pl.col("name") != "node2vec")
        .with_columns(pl.lit(128).alias("dim"), p=plnone, q=plnone)
        .select(colnames)
    )

    df = pl.concat((df_low1, df_high1)).sort("index").drop("index")

    if outfile is not None:
        df.collect().write_parquet(outfile)

    return df
