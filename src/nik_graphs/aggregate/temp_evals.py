import inspect
import itertools
import zipfile
from pathlib import Path

DATASETS = ["cora", "computer", "photo", "citeseer", "mnist"]
TEMPERATURES = [x * 10**i for i in range(-4, 1) for x in [1, 5]]
RANDOM_STATES = [None, 1111, 2222]
N_EPOCHS = 30


# example plotname = "temperatures"
def deplist(dispatch):
    depvals = deps(dispatch).values()
    return [f for lis in depvals for f in lis]


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "temp_evals"

    dataset = "mnist"

    from ..modules.cne import tsimcne_nonparam

    sig = inspect.signature(tsimcne_nonparam)
    default_temp = sig.parameters["temp"].default
    # default_loss = sig.parameters["loss"].default
    default_n_epochs = sig.parameters["n_epochs"].default

    n_epochs = f",n_epochs={N_EPOCHS}" if default_n_epochs != N_EPOCHS else ""

    paths = []
    for dataset, temp, r in iterator():
        path = Path("../runs") / dataset
        tempstr = f",temp={temp}" if temp != default_temp else ""
        randstr = f",random_state={r}" if r is not None else ""
        paths.append(
            path / ("cne,metric=cosine" + n_epochs + tempstr + randstr)
        )

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
                    with zf.open("scores.csv") as f:
                        df1 = pl.read_csv(f).drop("step")

                    with zf.open("score.txt") as f:
                        score = float(f.read())
                        df2 = pl.DataFrame(
                            dict(epoch=[N_EPOCHS - 1], score=[score])
                        )
                # check if the final accuracy has already been
                # computed because the last embedding has been saved
                # (n_epochs % callback_freq == 0) in the training.
                if df2["epoch"].item() in df1["epoch"]:
                    df__ = df1
                else:
                    df__ = df1.vstack(df2)
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
            df.write_csv(f)

    return df
