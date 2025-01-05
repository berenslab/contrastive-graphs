import itertools
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
MODELDICT = {
    "CNE cosine": "cne,metric=cosine",
    "CNE cosine (τ=0.05)": "cne,metric=cosine,temp=0.05",
    "CNE euclidean": "cne",
    "CNE euclidean (τ=0.05)": "cne,temp=0.05",
    "CNEτ cosine": "cne,metric=cosine,loss=infonce-temp",
    "CNEτ euclidean": "cne,loss=infonce-temp",
    "CNE 2D": "cne,dim=2",
    "CNEτ 2D": "cne,dim=2,loss=infonce-temp",
    "tsne": "tsne",
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
        k: [p / k / "1.zip" for p in paths]
        for k in [".", "..", "lin", "knn", "recall"]
    }
    return depdict


def iterator():
    return itertools.product(DATASETS, MODELDICT.items(), RANDOM_STATES)


def aggregate_path(path, outfile=None):
    import zipfile
    from collections import defaultdict

    import numpy as np
    import polars as pl

    results = defaultdict(list)
    for key, ziplist in deps(path).items():
        for (dataset, (mname, modelstr), r), zipf in zip(iterator(), ziplist):
            zpath = zipfile.Path(zipf)

            # "." is the first entry, so here we store all columns
            if key == ".":
                # hacky way to get the last temperature value when we
                # make this parameter learnable
                if ",loss=infonce-temp" in modelstr:
                    with (zpath / "lightning_logs/metrics.csv").open() as f:
                        train_df = pl.read_csv(f)
                        temp = (
                            train_df["logtemp"]
                            .drop_nulls()
                            .tail(1)
                            .exp()
                            .item()
                        )
                    results["learned_temp"].append(temp)
                else:
                    results["learned_temp"].append(None)
                results["dataset"].append(dataset)
                results["name"].append(mname)
                results["run_name"].append(modelstr)
                results["random_state"].append(r)
                results["n_epochs"].append(N_EPOCHS)
                secs = (zpath / "elapsed_secs.txt").read_text()
                results["time"].append(float(secs))
            elif key == "..":
                npz = np.load(zipf)
                n, m = npz["shape"]
                nnz = npz["data"].shape[0]
                results["n_pts"].append(n)
                results["n_edges"].append(nnz)
            else:
                acctxt = (zpath / "score.txt").read_text()
                results[key].append(float(acctxt))

    df = pl.DataFrame(results)
    if outfile is not None:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
