from ..plot import name2color, translate_plotname
from ..tex_utils import IndentedWriter


def deplist(dispatch=None, format="txt"):

    if format == "tex":
        import inspect
        from pathlib import Path

        projroot = Path(__file__).parent.parent.parent.parent
        tdir = projroot / "media/table"
        d = [
            inspect.getfile(IndentedWriter),
            inspect.getfile(name2color),
            tdir / "icml2025.sty",
            tdir / "icml2025.bst",
        ]
    else:
        d = []

    return [
        "../../dataframes/all_benchmarks.parquet",
        "../../dataframes/main_benchmarks.parquet",
    ] + d


def format_table(dispatch, outfile, format="tex"):
    import polars as pl

    df = pl.read_parquet(deplist(format=format)[0])
    if format == "tex":
        df_n2v = pl.read_parquet(deplist(format=format)[1]).filter(
            pl.col("name") == "node2vec"
        )
        idx = df.with_row_index().select("index", "name")
        df = (
            pl.concat([df.filter(pl.col("name") != "node2vec"), df_n2v])
            .join(idx, on="name")
            .sort("index")
            .drop("index")
        )

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
        df.sort("n_edges")
        .group_by(["dataset", "name", "dim"], maintain_order=True)
        .agg(
            pl.col("knn").map_elements(mean_std_fmt, return_dtype=str),
            pl.col("lin").map_elements(mean_std_fmt, return_dtype=str),
            pl.col("recall").map_elements(mean_std_fmt, return_dtype=str),
            pl.col("time").map_elements(mean_std_fmt_time, return_dtype=str),
        )
    )
    # df_fmt looks something like:
    #
    # ┌──────────┬──────────────────┬────────────┬───┬─────────────┐
    # │ dataset  ┆ name             ┆ knn        ┆   ┆ time        │
    # ╞══════════╪══════════════════╪════════════╪═══╪═════════════╡
    # │ cora     ┆ CNE (τ=0.5)      ┆ 77.96±1.0% ┆   ┆ 0.01±0.0 hr │
    # │ cora     ┆ deepwalk         ┆ 77.78±0.4% ┆ … ┆ 0.06±0.0 hr │
    # │ computer ┆ CNE (τ=0.5)      ┆ 90.81±0.4% ┆   ┆ 0.15±0.0 hr │
    # │ computer ┆ CNE (τ=0.05)     ┆ 91.50±0.2% ┆   ┆ 0.14±0.0 hr │
    #                                 ⋮

    match format:
        case "tex":
            return tex_table(df, outfile=outfile)
        case "txt":
            return txt_table(df_fmt, outfile=outfile)
        case "parquet":
            df_fmt.write_parquet(outfile)
            return df_fmt
        case _:
            raise ValueError(
                f"{format=!r} is not valid for formatting the table"
            )


def tex_table(df, outfile, metric_keys=["knn", "lin", "recall"]):
    import re

    import matplotlib as mpl
    import polars as pl

    df = (
        df.sort("n_edges")
        .group_by(["dataset", "name", "dim"], maintain_order=True)
        .agg(
            pl.col("knn", "lin", "recall").mean().name.prefix("mean_"),
            pl.col("knn", "lin", "recall").std().name.prefix("std_"),
        )
    )

    def mean_std_fmt_tex(df, metric):
        return (
            df.with_columns(
                (
                    pl.col(f"mean_{metric}")
                    >= pl.col(f"mean_{metric}").max().over("dataset", "dim")
                    - 0.01
                ).alias(f"bold_{metric}")
            )
            .with_columns(
                pl.format(
                    "{{}{}}±{}",
                    pl.when(pl.col(f"bold_{metric}"))
                    .then(pl.lit(r"\bf"))
                    .otherwise(pl.lit("")),
                    (pl.col(f"mean_{metric}") * 100).round(1),
                    (pl.col(f"std_{metric}") * 100).round(1),
                ).alias(metric)
            )
            .drop(f"bold_{metric}", f"mean_{metric}", f"std_{metric}")
        )

    for metric in metric_keys:
        df = df.pipe(mean_std_fmt_tex, metric=metric)

    n_dims = len(df.unique("dim"))
    datasets = df["dataset"].unique()
    begintable = r"\begin{table*}[t]"
    begintabular = (
        r"\begin{tabular}"
        f"{{{'' if n_dims < 2 else 'c'}l{'r' * len(datasets)}}}"
    )
    endtabular = r"\end{tabular}"
    endtable = r"\end{table*}"

    def tr(x):

        # need to double-escape in re.sub because it interprets
        # e.g. \(, so we need to add the second backslash.
        x = x.replace("±", r"${}\pm{}$")
        colorf = (
            r"{{\bf\color{{gne}}{x}}}"
            if x in ["tsne", "cne,temp=0.05", "cne,loss=infonce-temp"]
            else "{x}"
        )
        x = translate_plotname(x, _return="identity")
        x = re.sub("τ(.*)", r"$\\tau\1$", x)
        return colorf.format(x=x)

    def tex_table_center(s):
        return r"\vadjust{}\hfill{}" f"{s}" r"\hfill\vadjust{}"

    with open(outfile, "x") as f:
        fw = IndentedWriter(f)
        fw.writeln(r"\documentclass{article}")
        fw.writeln(r"\usepackage{booktabs}")
        fw.writeln(r"\usepackage{multirow}")
        fw.writeln(r"\usepackage{icml2025}")
        fw.writeln(r"\usepackage[T1]{fontenc}")

        fw.writeln(r"\begin{document}")
        fw.writeln(begintable)
        with fw.indent():
            color_hex = mpl.colors.to_hex(name2color("tsne"))[1:].upper()
            fw.writeln(r"\definecolor{gne}{HTML}" f"{{{color_hex}}}")
            for key in metric_keys:
                # write out a header comment showing the knn/lin/...
                fw.writeln("%" * 20 + f"\n%%%{key:^14s}%%%\n" + "%" * 20)
                fw.writeln(rf"\caption{{{key} accuracy table.}}")
                fw.writeln(r"\vskip0.075in")
                fw.writeln(r"\small\centering")
                fw.writeln(begintabular)
                with fw.indent():
                    fw.writeln(r"\toprule")

                    fw.writeln(
                        " &\n    ".join(
                            ([] if n_dims < 2 else ["$d$"])
                            + [
                                tex_table_center("Method"),
                            ]
                            + df.unique("dataset", maintain_order=True)
                            .select(
                                pl.col("dataset").map_elements(
                                    translate_plotname, return_dtype=str
                                )
                            )
                            .to_series()
                            .map_elements(
                                lambda row: tex_table_center(row).replace(
                                    # rename "MNIST $k$NN" to just "MNIST"
                                    " $k$NN",
                                    "",
                                ),
                                return_dtype=str,
                            )
                            .to_list(),
                        )
                        + r" \\"
                    )

                    for (dim,), df_ in df.group_by("dim", maintain_order=True):
                        fw.writeln(r"\midrule")

                        df1 = df_.pivot("dataset", index="name", values=key)

                        if n_dims >= 2:
                            fw.writeln(
                                r"\multirow" f"{{{len(df1)}}}{{*}}{{{dim}}}"
                            )

                        for row in df1.rows():
                            fw.write("& ")
                            fw.writeln(" & ".join(tr(r) for r in row) + r" \\")

                    fw.writeln(r"\bottomrule")
                fw.writeln(endtabular)
        fw.writeln(endtable)
        fw.writeln(r"\end{document}")


def txt_table(df, outfile=None):
    import polars as pl

    if outfile is not None:
        with open(outfile, "x") as f:
            with pl.Config(
                tbl_cols=-1,
                tbl_rows=-1,
                tbl_hide_column_data_types=True,
                tbl_hide_dataframe_shape=True,
            ) as cfg:
                cfg.set_tbl_formatting(rounded_corners=True)
                print(df, file=f)
