import inspect
import itertools
import zipfile
from pathlib import Path

DATASETS = [
    "cora",
    "computer",
    "photo",
    "citeseer",
    "mnist",
    "arxiv",
    "pubmed",
    "mag",
]
TEMPERATURES = [x * 10**i for i in range(-4, 1) for x in [1, 5]]
RANDOM_STATES = [None, 1111, 2222]


# example plotname = "temperatures"
def deplist(dispatch):
    depvals = deps(dispatch).values()
    return [f for lis in depvals for f in lis]


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "temp_evals"

    from ..modules.cne import tsimcne_nonparam

    sig = inspect.signature(tsimcne_nonparam)
    default_temp = sig.parameters["temp"].default

    paths = []
    for dataset, temp, r in iterator():
        path = Path("../runs") / dataset
        tempstr = f",temp={temp}" if temp != default_temp else ""
        randstr = f",random_state={r}" if r is not None else ""
        paths.append(path / ("cne" + tempstr + randstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in [".", "lin", "knn", "recall"]
    }
    return depdict


def iterator():
    return itertools.product(DATASETS, TEMPERATURES, RANDOM_STATES)


def aggregate_path(path, outfile=None):
    import polars as pl

    depd = deps(path)

    # labels = np.load(files["data"])["labels"]
    df_dict = dict()
    for k, v in depd.items():
        df_ = None
        # same iteration scheme as in `deps()` above so that the order
        # between the zipfile and the parameters match.
        for (dataset, temp, r), zipf in zip(iterator(), v):
            if k == ".":
                # read the loss from the run lightning_logs/metrics.csv
                with zipfile.ZipFile(zipf) as zf:
                    with zf.open("lightning_logs/metrics.csv") as f:
                        df_metrics = pl.read_csv(f)
                dfg = df_metrics.group_by("epoch", maintain_order=True)
                df__ = dfg.mean()[["epoch", "loss"]]

            else:
                with zipfile.ZipFile(zipf) as zf:
                    with zf.open("score.txt") as f:
                        score = float(f.read())
                        df__ = pl.DataFrame(dict(epoch=[99], score=[score]))
            df__ = df__.with_columns(
                pl.lit(dataset).alias("dataset"),
                pl.lit(float(temp)).alias("temp"),
                pl.lit(r, dtype=pl.Int32).alias("random_state"),
            ).rename(dict(score=k), strict=False)
            df_ = df_.vstack(df__) if df_ is not None else df__
        df_dict[k] = df_

    # take any of the dfs as initial value
    df = df_dict.pop(".")
    for k, df_ in df_dict.items():
        df = df.join(
            df_,
            on=["temp", "epoch", "random_state", "dataset"],
            join_nulls=True,
        )

    if outfile is not None:
        # with zipfile.ZipFile(outfile, "x") as zf:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
