import inspect
import itertools
from collections import defaultdict
from pathlib import Path

DATASETS = [
    "cora",
    "pubmed",
    "computer",
    "photo",
    "citeseer",
    "mnist",
    "arxiv",
    "mag",
]
RANDOM_STATES = [None, 1111, 2222]
N_EPOCHS = 100


def deplist(dispatch):
    depvals = deps(dispatch).values()
    return [f for lis in depvals for f in lis]


def deps(dispatch):
    assert str(dispatch) == "learned_temp"

    from ..modules.cne import tsimcne_nonparam

    sig = inspect.signature(tsimcne_nonparam)
    default_loss = sig.parameters["loss"].default
    default_metric = sig.parameters["metric"].default
    default_n_epochs = sig.parameters["n_epochs"].default

    mname = "cosine"
    metric = f",metric={mname}" if default_metric != mname else ""
    lname = "infonce-temp"
    loss = f",loss={lname}" if default_loss != lname else ""
    n_epochs = f",n_epochs={N_EPOCHS}" if default_n_epochs != N_EPOCHS else ""

    paths = []
    for dataset, r in iterator():
        path = Path("../runs") / dataset
        randstr = f",random_state={r}" if r is not None else ""
        paths.append(path / ("cne" + metric + loss + n_epochs + randstr))

    return [p / "1.zip" for p in paths]


def iterator():

    return itertools.product(DATASETS, RANDOM_STATES)


def aggregate_path(path, outfile=None):
    import zipfile

    import polars as pl

    df_dict = defaultdict(list)
    for (dataset, r), zipf in zip(iterator(), deplist(path)):
        zpath = zipfile.Path(zipf)

        with (zpath / "lightning_logs/metrics.csv").open() as f:
            train_df = pl.read_csv(f)

        temp = train_df.select("logtemp").tail(1).exp().item()

        df_dict["temp"].append(temp)
        df_dict["dataset"].append(dataset)
        df_dict["random_state"].append(r)

    df = pl.DataFrame(df_dict)
    if outfile is not None:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
