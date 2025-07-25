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
LAYOUTS = [
    "tsne",
    "cne,dim=2,metric=euclidean",
    "cne,temp=0.05,dim=2,metric=euclidean",
    "cne,loss=infonce-temp,dim=2,metric=euclidean",
    "spectral",
    "sgtsnepi",
    "drgraph",
    "fa2",
    "tfdp",
    "graphmae",
    "nmf",
]

RANDOM_STATES = [None, 1111, 2222]


def deplist(dispatch: Path):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "low_dim_benchmarks"

    paths = []

    for dataset, mname, r in iterator():
        # modname = model.split(",")[0]

        path = Path("../runs") / dataset
        randstr = f",random_state={r}" if r is not None else ""
        paths.append(path / (mname + randstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in [".", "..", "lin", "knn", "recall", "lpred", "spcorr"]
    }
    return depdict


def iterator():
    return itertools.product(DATASETS, LAYOUTS, RANDOM_STATES)


def aggregate_path(path, outfile=None):
    import zipfile
    from collections import defaultdict

    import numpy as np
    import polars as pl

    results = defaultdict(list)
    pathdict = dict()
    for key, ziplist in deps(path).items():
        for (dataset, mname, r), zipf in zip(iterator(), ziplist):
            zpath = zipfile.Path(zipf)

            # "." is the first entry, so here we store all columns
            if key == ".":
                results["dataset"].append(dataset)
                results["name"].append(mname)
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

    order = (
        "tsne cne,temp=0.05,dim=2,metric=euclidean cne,dim=2,metric=euclidean "
        "cne,loss=infonce-temp,dim=2,metric=euclidean "
        "sgtsnepi drgraph fa2 tfdp spectral graphmae nmf"
    ).split()
    assert all(x in order for x in LAYOUTS)
    df = (
        pl.DataFrame(results)
        .join(pl.DataFrame(dict(name=order)).with_row_index(), on="name")
        .sort("index")
        .drop("index")
    )

    if outfile is not None:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
