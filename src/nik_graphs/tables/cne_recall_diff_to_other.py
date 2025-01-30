def deplist(dispatch=None, format="txt"):

    return ["../../dataframes/main_benchmarks.parquet"]


def format_table(dispatch, outfile, format="tex"):
    import polars as pl

    df = pl.read_parquet(deplist()[0])

    rdiffs = []
    for (g,), df_ in df.group_by("dataset"):
        df1 = df_.group_by("name").agg(pl.mean("recall")).top_k(2, by="recall")
        recall_diff = (
            df1.filter(pl.col("name") == "cne,temp=0.05")["recall"]
            - df1.filter(pl.col("name") != "cne,temp=0.05")["recall"]
        ).item()
        rdiffs.append(recall_diff)

    mean = pl.Series(rdiffs).mean()

    with open(outfile, "x") as f:
        f.write(
            "Across all datasets, the average gap in neighbor recall "
            "between graph CNE and the best other method was "
            f"{mean*100:.1f} percentage points. "
        )

        f.write(f"\n\n\n{mean}\n")
