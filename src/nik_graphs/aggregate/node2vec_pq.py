import itertools
from pathlib import Path

DATASETS = [
    "cora",
    "computer",
    "photo",
    "citeseer",
    "mnist",
    "pubmed",
    "arxiv",
    "mag",
]
RANDOM_STATES = [None, 1111, 2222]
PS = [0.25, 0.5, 1, 2, 4]
QS = [0.25, 0.5, 1, 2, 4]


# example plotname = "temperatures"
def deplist(dispatch: Path):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    paths = []

    for dataset, p, q, r in iterator():
        # modname = model.split(",")[0]

        path = Path("../runs") / dataset
        randstr = f",random_state={r}" if r is not None else ""
        pstr = "" if p == 1 else f",{p=}"
        qstr = "" if q == 1 else f",{q=}"
        modelstr = f"node2vec{pstr}{qstr}"
        paths.append(path / (modelstr + randstr))

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in [
            ".",
            "..",
            "lin",
            "knn,metric=cosine",
            "recall,metric=cosine",
            "lpred,metric=cosine",
            "spcorr,metric=cosine",
        ]
    }
    return depdict


def iterator():
    return itertools.product(DATASETS, PS, QS, RANDOM_STATES)


def aggregate_path(path, outfile=None):
    import zipfile
    from collections import defaultdict

    import numpy as np
    import polars as pl

    results = defaultdict(list)
    pathdict = dict()
    for key, ziplist in deps(path).items():
        for (dataset, p, q, r), zipf in zip(iterator(), ziplist):
            zpath = zipfile.Path(zipf)

            # "." is the first entry, so here we store all columns
            if key == ".":
                results["dataset"].append(dataset)
                results["name"].append("node2vec")
                results["p"].append(p)
                results["q"].append(q)
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
                shortkey = key.replace(",metric=cosine", "")
                results[shortkey].append(float(acctxt))

    df = pl.DataFrame(results)
    if outfile is not None:
        with open(outfile, "xb") as f:
            df.write_parquet(f)

    return df
