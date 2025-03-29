import itertools
from pathlib import Path

DATASETS = [
    "citeseer",
    "cora",
    "photo",
    "computer",
    "pubmed",
    "mnist",
]


# example plotname = "temperatures"
def deplist(dispatch: Path):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    paths = []

    for dataset in DATASETS:

        path = Path("../runs") / dataset
        modelstr = "gfeat"
        paths.append(path / (modelstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in [
            ".",
            "..",
            "lin",
            "knn,metric=cosine",
            "recall,metric=cosine",
            "lpred,metric=cosine",
        ]
    }
    return depdict


def aggregate_path(path, outfile=None):
    import zipfile
    from collections import defaultdict

    import numpy as np
    import polars as pl

    results = defaultdict(list)
    pathdict = dict()
    for key, ziplist in deps(path).items():
        for dataset, zipf in zip(DATASETS, ziplist):
            zpath = zipfile.Path(zipf)

            # "." is the first entry, so here we store all columns
            if key == ".":
                results["dataset"].append(dataset)
                results["name"].append("gfeat")
                results["random_state"].append(None)
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
                shortkey = key.replace(",metric=cosine", "")
                results[shortkey].append(float(acctxt))

    df = pl.DataFrame(results).cast(dict(random_state=int))
    if outfile is not None:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
