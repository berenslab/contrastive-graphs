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
    "CNE, τ=0.5": "cne",
    "CNE, τ=0.05": "cne,temp=0.05",
    "CNEτ": "cne,loss=infonce-temp",
    "deepwalk": "deepwalk",
    "node2vec": "node2vec",
}
LAYOUTDICT = {
    k: k for k in ["tsne", "spectral", "sgtsnepi", "drgraph", "fa2", "tfdp"]
}
MODELS = [
    "tsne",
    "spectral",
    "sgtsnepi",
    "drgraph",
    "fa2",
    "tfdp",
    "deepwalk",
    "node2vec",
    "cne",
    "cne,temp=0.05",
    "cne,loss=infonce-temp",
]
RANDOM_STATES = [None, 1111, 2222]


# example plotname = "temperatures"
def deplist(dispatch: Path):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "high_dim_benchmarks"

    paths = []

    for dataset, (mname, modelstr), r in iterator():
        # modname = model.split(",")[0]

        path = Path("../runs") / dataset
        randstr = f",random_state={r}" if r is not None else ""
        paths.append(path / (modelstr + randstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in [".", "..", "lin", "knn", "recall"]
    }
    return depdict


def iterator():
    return (
        row
        for row in itertools.product(
            DATASETS, MODELDICT.items(), RANDOM_STATES
        )
        if not (row[0] == "mag" and row[1][1] == "cne" and row[2] == 1111)
    )


def aggregate_path(path, outfile=None):
    import zipfile
    from collections import defaultdict

    import numpy as np
    import polars as pl

    results = defaultdict(list)
    pathdict = dict()
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
                secs = (zpath / "elapsed_secs.txt").read_text()
                results["time"].append(float(secs))
            elif key == "..":
                path = zipf.resolve()
                if path not in pathdict:
                    npz = np.load(zipf)
                    n, m = npz["shape"]
                    nnz = npz["data"].shape[0]
                    results["n_pts"].append(n)
                    results["n_edges"].append(nnz)
                    pathdict[path] = n, nnz
                else:
                    n, nnz = pathdict[path]
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
