def deplist(dispatch=None):

    return ["../../dataframes/high_dim_benchmarks.parquet"]


def format_table(dispatch, outfile, format="tex"):
    import polars as pl

    df = pl.read_parquet(deplist()[0])

    def mean_std_fmt(s: pl.Series) -> str:
        mean = s.mean()
        std = s.std()
        return f"{mean * 100:5.2f}±{std:3.1%}"

    df_fmt = df.group_by(["dataset", "name"], maintain_order=True).agg(
        pl.col("knn").map_elements(mean_std_fmt, return_dtype=str),
        pl.col("lin").map_elements(mean_std_fmt, return_dtype=str),
        pl.col("recall").map_elements(mean_std_fmt, return_dtype=str),
    )
    # df_fmt looks something like:
    #
    # shape: (8, 5)
    # ┌──────────┬──────────┬────────────┬────────────┬────────────┐
    # │ dataset  ┆ name     ┆ knn        ┆ lin        ┆ recall     │
    # │ ---      ┆ ---      ┆ ---        ┆ ---        ┆ ---        │
    # │ str      ┆ str      ┆ str        ┆ str        ┆ str        │
    # ╞══════════╪══════════╪════════════╪════════════╪════════════╡
    # │ cora     ┆ CNE      ┆ 76.76±1.1% ┆ 79.02±0.6% ┆ 65.16±0.9% │
    # │ cora     ┆ CNEτ     ┆ 75.70±0.6% ┆ 75.70±0.4% ┆ 79.86±0.3% │
    # │ cora     ┆ deepwalk ┆ 77.96±0.7% ┆ 79.31±0.2% ┆ 63.22±0.8% │
    # │ cora     ┆ node2vec ┆ 40.40±3.1% ┆ 55.08±0.5% ┆  1.01±0.1% │
    # │ computer ┆ CNE      ┆ 90.51±0.5% ┆ 90.28±0.7% ┆ 28.20±0.5% │
    # │ computer ┆ CNEτ     ┆ 92.38±0.1% ┆ 89.31±0.5% ┆ 45.04±0.4% │
    # │ computer ┆ deepwalk ┆ 91.45±0.3% ┆ 88.64±0.9% ┆ 30.22±0.1% │
    # │ computer ┆ node2vec ┆ 87.67±0.8% ┆ 85.60±0.6% ┆ 15.04±0.4% │
    # └──────────┴──────────┴────────────┴────────────┴────────────┘

    match format:
        case "tex":
            return tex_table(df_fmt, outfile=outfile)
        case "txt":
            return txt_table(df_fmt, outfile=outfile)
        case _:
            raise ValueError(
                f"{format=!r} is not valid for formatting the table"
            )


def tex_table(df, outfile=None): ...


def txt_table(df, outfile=None):
    import polars as pl

    if outfile is not None:
        with open(outfile, "x") as f:
            with pl.Config(
                tbl_cols=-1,
                tbl_rows=-1,
                tbl_hide_column_data_types=True,
                tbl_hide_dataframe_shape=True,
            ):
                print(df, file=f)
