def deplist(dispatch=None):

    return ["../../dataframes/high_dim_benchmarks.parquet"]


def format_table(dispatch, outfile, format="tex"):
    import polars as pl

    df = pl.read_parquet(deplist()[0])

    def mean_std_fmt(s: pl.Series) -> str:
        mean = s.mean()
        std = s.std()
        return f"{mean * 100:5.2f}±{std:3.1%}"

    def mean_std_fmt_time(s: pl.Series) -> str:
        hrs = s / (60 * 60)
        mean = hrs.mean()
        std = hrs.std()
        return f"{mean:.2f}±{std:.1f} hr"

    df_fmt = (
        df.group_by(["dataset", "run_name"], maintain_order=True)
        .agg(
            pl.col("name").first(),
            pl.col("knn").map_elements(mean_std_fmt, return_dtype=str),
            pl.col("lin").map_elements(mean_std_fmt, return_dtype=str),
            pl.col("recall").map_elements(mean_std_fmt, return_dtype=str),
            pl.col("time").map_elements(mean_std_fmt_time, return_dtype=str),
        )
        .drop("run_name")
    )
    # df_fmt looks something like:
    #
    # ┌──────────┬──────────────────┬────────────┬───┬─────────────┐
    # │ dataset  ┆ name             ┆ knn        ┆   ┆ time        │
    # ╞══════════╪══════════════════╪════════════╪═══╪═════════════╡
    # │ cora     ┆ CNE (τ=0.5)      ┆ 77.96±1.0% ┆   ┆ 0.01±0.0 hr │
    # │ cora     ┆ CNE (̂τ = 0.081)  ┆ 75.63±0.5% ┆   ┆ 0.01±0.0 hr │
    # │ cora     ┆ deepwalk         ┆ 77.78±0.4% ┆ … ┆ 0.06±0.0 hr │
    # │ computer ┆ CNE (τ=0.5)      ┆ 90.81±0.4% ┆   ┆ 0.15±0.0 hr │
    # │ computer ┆ CNE (τ=0.05)     ┆ 91.50±0.2% ┆   ┆ 0.14±0.0 hr │
    # │ computer ┆ CNE ((̂τ = 0.077) ┆ 91.60±0.3% ┆   ┆ 0.15±0.0 hr │
    #                                 ⋮

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
