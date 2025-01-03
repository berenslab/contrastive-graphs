from pathlib import Path

COLORS = ["#820008", "#be241c", "#eb7135", "#1c69ff", "#007100"]
# ["#0414d2", "#0061f7", "#61aaf7", "#c23900", "#a20075"]


def deplist(dispatch: Path):
    return ["../dataframes/high_dim_benchmarks.parquet"]


def plot_path(plotname, outfile, format="pdf"):
    import polars as pl

    df = pl.read_parquet(deplist(plotname)[0])

    return plot(df, outfile=outfile, format=format)


def plot(df_full, outfile, format="pdf"):
    import matplotlib as mpl
    import polars as pl
    from matplotlib import pyplot as plt

    from ..plot import add_letters

    keys = ["knn", "lin", "recall"]  # , "time"]

    fig, axs = plt.subplots(
        1, 3, figsize=(3.25, 1), sharex=True, squeeze=False
    )
    for i, (key, ax) in enumerate(zip(keys, axs.flat)):
        df_metric = df_full.group_by(
            ["dataset", "run_name"], maintain_order=True
        ).agg(
            pl.first("name", "n_edges"),
            pl.mean(key).alias("mean"),
            pl.std(key).alias("std"),
        )
        for color, ((_,), df) in zip(
            COLORS, df_metric.group_by("run_name", maintain_order=True)
        ):
            df = df.sort(by="n_edges")
            label = df["name"].head(1).item()
            x, m, std = df[["n_edges", "mean", "std"]]
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
            (line,) = ax.plot(x, m, label=label, marker="o", color=color)
            ax.fill_between(
                x, m + std, m - std, color=line.get_color(), alpha=0.618
            )
        if key == "lin":
            add_dataset_names(ax, df, key)
        ax.set(title=key, xscale="log")
        if i == 0:
            ax.legend(fontsize="x-small")

    [ax.set_xlabel("number of edges") for ax in axs[-1]]
    add_letters(axs.flat)
    fig.savefig(outfile, format=format)


def add_dataset_names(ax, df1, ykey):
    # loosely based on
    # https://datascience.stackexchange.com/questions/66506/
    # what-is-a-good-method-for-detecting-local-minims-and-maxims
    # sdiff = df1["mean"].diff()
    # sign_ind = sdiff.tail(-1).sign() - sdiff.head(-1).sign()
    for (dataset,), df in df1.group_by("dataset"):

        # index, dataset, n_edges, mean, std = row
        # import sys

        # print(df, file=sys.stderr)

        # all of those alginment and xytext values have been
        # determined manually by looking at the plot
        match dataset:
            case "cora":
                m, std = df[["mean", "std"]].max()
                y = m + std
                kwargs = dict(ha="left", va="bottom", xytext=(-1.5, 1.75))
            case "citeseer":
                m, std = df[["mean", "std"]].min()
                y = m - std
                kwargs = dict(ha="left", va="baseline", xytext=(2.25, 0))
            case "computer":
                m, std = df[["mean", "std"]].min()
                y = m - std
                kwargs = dict(ha="left", va="top", xytext=(-3, 0))
            case "photo":
                m, std = df[["mean", "std"]].max()
                y = m + std
                kwargs = dict(ha="right", va="center", xytext=(-1.5, 0))
            case "mnist":
                m, std = df[["mean", "std"]].max()
                y = m + std
                kwargs = dict(ha="right", va="baseline", xytext=(-1.5, 0))
            case _:
                import warnings

                warnings.warn(f"Uknown {dataset=!r}, default annotation")
                y = df[["mean"]].mean()
                kwargs = dict(xytext=(0, 0))

        x = df["n_edges"].head(1).item()
        ax.annotate(
            text=dataset,
            xy=(x, y.item()),
            **kwargs,
            textcoords="offset points",
            fontsize="small",
        )
