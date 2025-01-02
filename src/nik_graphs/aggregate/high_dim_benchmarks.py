import itertools
from pathlib import Path

DATASETS = ["cora", "computer", "photo", "citeseer", "mnist"]
MODELDICT = dict(
    CNE="cne,metric=cosine",
    CNEτ="cne,metric=cosine,loss=infonce-temp",
    deepwalk="deepwalk",
    node2vec="node2vec",
)
N_EPOCHS = 10
RANDOM_STATES = [None, 1111, 2222]


# example plotname = "temperatures"
def deplist(dispatch: Path):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "temp_evals"

    paths = []

    # the default n_epochs for cne, node2vec, deepwalk is 100
    n_epochs = f",n_epochs={N_EPOCHS}" if 100 != N_EPOCHS else ""

    for dataset, (mname, modelstr), r in iterator():
        # modname = model.split(",")[0]

        path = Path("../runs") / dataset
        randstr = f",random_state={r}" if r is not None else ""
        paths.append(path / (modelstr + n_epochs + randstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in [".", "lin", "knn", "recall"]
    }
    return depdict


def iterator():
    return itertools.product(DATASETS, MODELDICT.items(), RANDOM_STATES)


def aggregate_path(path, outfile=None):
    import zipfile

    import polars as pl
    import yaml

    to_drop = "dof ta ru val_logtemp val_ru val_ta val_loss".split()

    df_dict = dict()
    for key, ziplist in deps(path).items():
        for (dataset, (mname, modelstr), r), zipf in zip(iterator(), ziplist):
            zpath = zipfile.Path(zipf)

            if key == ".":
                with (zpath / "lightning_logs/metrics.csv").open() as f:
                    train_df = pl.read_csv(f)
                with (zpath / "lightning_logs/hparams.yaml").open() as f:
                    hparams = yaml.safe_load(f)

                batches_per_epoch = hparams["batches_per_epoch"]
                df_ = train_df.drop(to_drop).drop_nulls()
                df_ = df_.with_columns(
                    pl.lit(batches_per_epoch).alias("batches_per_epoch"),
                    pl.lit(dataset).alias("dataset"),
                )
                # this key "." comes first, so we can simply store the
                # df in the dictionary.
                df_dict[dataset] = df_

            else:
                with (zpath / "scores.csv").open() as f:
                    df_ = pl.read_csv(f)
                # subtract 1 from the step so it aligns with the steps
                # in train_df
                df_ = (
                    df_.select(pl.all(), s=pl.col("step") - 1)
                    .drop("step")
                    .rename(dict(s="step", score=key))
                )
                df_ = df_.with_columns(
                    pl.lit(dataset).alias("dataset"),
                    # pl.lit(r, dtype=pl.Int32).alias("random_state"),
                )

                # rope in the intermediate values and join them onto the df
                df_dict[dataset] = df_dict[dataset].join(
                    df_,
                    on=["dataset", "epoch", "step"],
                    how="outer",
                    coalesce=True,
                )

    df = pl.concat([df for df in df_dict.values()])
    if outfile is not None:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
