from collections import defaultdict
from pathlib import Path

from ..plot import translate_plotname
from ..tex_utils import IndentedWriter

DATASETS = [
    "mnist",
    "cora",
    "citeseer",
    "pubmed",
    "photo",
    "computer",
    "arxiv",
    "mag",
]


def deplist(dispatch=None, format="txt"):

    if format == "tex":
        import inspect

        projroot = Path(__file__).parent.parent.parent.parent
        tdir = projroot / "media/table"
        d = [
            inspect.getfile(IndentedWriter),
            inspect.getfile(translate_plotname),
            tdir / "icml2025.sty",
            tdir / "icml2025.bst",
        ]
    else:
        d = []

    if format == "parquet":
        path = Path("../../runs")
        return [path / ds / "1.zip" for ds in DATASETS] + d
    else:
        return ["dataset_info.parquet"] + d


def format_table(dispatch, outfile, format="tex"):
    import polars as pl

    deps = deplist(format=format)
    if format == "parquet":
        df = assemble_df(deps)
    else:
        df = pl.read_parquet(deps[0])

    match format:
        case "tex":
            return tex_table(df, outfile=outfile)
        case "txt":
            return txt_table(df, outfile=outfile)
        case "parquet":
            df.write_parquet(outfile)
            return df
        case _:
            raise ValueError(
                f"{format=!r} is not valid for formatting the table"
            )


def assemble_df(deps):
    import numpy as np
    import polars as pl
    from scipy import sparse

    dic = defaultdict(list)
    for dataset, zipf in zip(DATASETS, deps):
        dic["dataset_key"].append(zipf.parent.name)
        name = translate_plotname(zipf.parent.name)
        dic["name"].append(name)
        A = sparse.load_npz(zipf)
        dic["n_pts"].append(A.shape[0])
        dic["n_edges"].append(A.nnz)

        n_labels = len(np.unique(np.load(zipf)["labels"]))
        dic["n_labels"].append(n_labels)

    return pl.DataFrame(dic).sort("n_edges")


def tex_table(df, outfile):
    import polars as pl

    df = df.with_columns(
        (pl.col("n_edges") / pl.col("n_pts")).alias("edge_pts_ratio")
    ).drop("dataset_key")

    begintable = r"\begin{table}[t]"
    begintabular = r"\begin{tabular}{lrrcc}"
    endtabular = r"\end{tabular}"
    endtable = r"\end{table}"

    with open(outfile, "x") as f:
        fw = IndentedWriter(f)
        fw.writeln(r"\documentclass{article}")
        fw.writeln(r"\usepackage{booktabs}")
        fw.writeln(r"\usepackage{icml2025}")
        fw.writeln(r"\usepackage[T1]{fontenc}")

        fw.writeln(r"\begin{document}")
        fw.writeln(begintable)
        with fw.indent():
            fw.writeln(r"\caption{Dataset information.}")
            fw.writeln(r"\vskip0.075in")
            fw.writeln(r"\centering")  # \small
            fw.writeln(begintabular)
            with fw.indent():
                fw.writeln(r"\toprule")

                tr_col = dict(
                    name=r"\vadjust{}\hfill Dataset\hfill\vadjust{}",
                    n_pts=r"\vadjust{}\hfill Nodes\hfill\vadjust{}",
                    n_edges=r"\vadjust{}\hfill Edges\hfill\vadjust{}",
                    n_labels="Classes",
                    edge_pts_ratio="$E/N$",
                )
                fw.writeln(" & ".join(tr_col[c] for c in df.columns) + r" \\")
                fw.writeln(r"\midrule")

                for row in df.rows():

                    def tr(x):
                        match x:
                            case str(x):
                                s = x
                            case int(x):
                                s = f"{x:3_d}".replace("_", r"\thinspace")
                                # the "Classes" entries need to be
                                # \phantom padded, they are the only
                                # ints that will possibly be shorter
                                # than 3 digits, so we can replace the
                                # " " pad with \phantom{0}.
                                s = s.replace(" ", r"\phantom{0}")

                            case float(x):
                                s = f"{x:4.1f}".replace(" ", r"\phantom{0}")

                        return s

                    row_tex = (tr(r) for r in row)
                    fw.writeln(" & ".join(row_tex) + r" \\")

                fw.writeln(r"\bottomrule")
            fw.writeln(endtabular)
        fw.writeln(endtable)
        fw.writeln(r"\end{document}")


def txt_table(df, outfile=None):
    import polars as pl

    df = df.with_columns(pl.col("name").replace("MNIST $k$NN", "MNIST kNN"))
    if outfile is not None:
        with open(outfile, "x") as f:
            with pl.Config(
                tbl_cols=-1,
                tbl_rows=-1,
                tbl_hide_column_data_types=True,
                tbl_hide_dataframe_shape=True,
                tbl_cell_numeric_alignment="RIGHT",
                thousands_separator=" ",
            ) as cfg:
                cfg.set_tbl_formatting(rounded_corners=True)
                print(df, file=f)
