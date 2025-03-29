def deplist(dispatch=None):
    return [f"{d}_dim_benchmarks.parquet" for d in ["low", "high"]] + [
        "node2vec_pq.parquet",
        "gfeat_benchmarks.parquet",
    ]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_low, df_high, df_n2v, df_gfeat = [pl.scan_parquet(f) for f in deplist()]
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
    n2v_ix = (
        dfix.filter(pl.col("name") == "node2vec")
        .select("index")
        .head(1)
        .collect()
        .item()
    )
    df_high1 = (
        dfix.filter(pl.col("name") != "node2vec")
        .with_columns(pl.lit(128).alias("dim"), p=plnone, q=plnone)
        .select(colnames)
    )

    df_n2v1 = df_n2v.with_columns(
        dim=pl.lit(128),
        learned_temp=plnone,
        index=pl.lit(n2v_ix, dtype=pl.UInt32),
    ).select(colnames)

    df_gfeat1 = df_gfeat.with_columns(
        dim=plnone.cast(pl.Int32),
        learned_temp=plnone,
        index=pl.lit(100, dtype=pl.UInt32),
        p=plnone,
        q=plnone,
    ).select(colnames)

    df = (
        pl.concat((df_low1, df_high1, df_n2v1, df_gfeat1))
        .sort("index")
        .drop("index")
    )

    if outfile is not None:
        df.collect().write_parquet(outfile)

    return df
