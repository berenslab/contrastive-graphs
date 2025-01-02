import itertools
from pathlib import Path

DATASETS = ["cora", "computer", "photo", "citeseer", "mnist"]
MODELDICT = {
    "CNE (τ=0.5)": "cne,metric=cosine",
    "CNE (τ=0.05)": "cne,metric=cosine,temp=0.05",
    "CNEτ": "cne,metric=cosine,loss=infonce-temp",
    "deepwalk": "deepwalk",
    "node2vec": "node2vec",
}
N_EPOCHS = 100
RANDOM_STATES = [None, 1111, 2222]


# example plotname = "temperatures"
def deplist(dispatch: Path):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "high_dim_benchmarks"

    paths = []

    # the default n_epochs for cne, node2vec, deepwalk is 100
    n_epochs = f",n_epochs={N_EPOCHS}" if 100 != N_EPOCHS else ""

    for dataset, (mname, modelstr), r in iterator():
        # modname = model.split(",")[0]

        path = Path("../runs") / dataset
        randstr = f",random_state={r}" if r is not None else ""
        paths.append(path / (modelstr + n_epochs + randstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths] for k in ["lin", "knn", "recall"]
    }
    return depdict


def iterator():
    return itertools.product(DATASETS, MODELDICT.items(), RANDOM_STATES)


def aggregate_path(path, outfile=None):
    import zipfile
    from collections import defaultdict

    import polars as pl

    results = defaultdict(list)
    for key, ziplist in deps(path).items():
        for (dataset, (mname, modelstr), r), zipf in zip(iterator(), ziplist):
            # "knn" is the first entry, so here we store all columns
            if key == "knn":
                results["dataset"].append(dataset)
                results["name"].append(mname)
                results["run_name"].append(modelstr)
                results["random_state"].append(r)
                results["n_epochs"].append(N_EPOCHS)

            zpath = zipfile.Path(zipf)
            acctxt = (zpath / "score.txt").read_text()
            acc = float(acctxt)
            results[key].append(acc)

    df = pl.DataFrame(results)
    if outfile is not None:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
