def deplist(dispatch=None):
    return ["all_benchmarks.parquet"]


def aggregate_path(path, outfile=None):
    import polars as pl

    df_in = pl.scan_parquet(deplist()[0])

    # get the best p, q parameter combination for node2vec and save
    # those off in df_pq, so that we can do a semi join on that.  It
    # also carries along the name (p, q as nulls) so that all other
    # methods are also preserved
    join_on = ["dataset", "name", "p", "q"]
    df_pq = (
        df_in.group_by(join_on)
        .agg(pl.mean("recall"))
        .filter(pl.col("recall") == pl.max("recall").over("dataset", "name"))
    )

    colnames = df_in.collect_schema().names()
    df = (
        df_in.unpivot(["recall", "knn", "lin"], index=colnames)
        .join(df_pq, on=join_on, join_nulls=True, how="semi")
        .drop("variable", "value")
        .filter(
            (pl.col("name") == "cne,temp=0.05")
            | ~pl.col("name").str.starts_with("cne"),
        )
    )

    if outfile is not None:
        df.collect().write_parquet(outfile)

    return df
