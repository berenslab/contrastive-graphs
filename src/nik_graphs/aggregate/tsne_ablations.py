import itertools
import zipfile
from pathlib import Path

DATASETS = [
    "cora",
    "computer",
    "photo",
    "citeseer",
    "pubmed",
    "mnist",
    "arxiv",
    # "mag",
]
LAYOUTS = ["tsne"]
VARIANTS = ["", ",rownorm=1", ",initialization=random"]


# example plotname = "temperatures"
def deplist(dispatch):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "random_inits"

    paths = []
    for dataset, layout, variant in itertools.product(
        DATASETS, LAYOUTS, VARIANTS
    ):
        paths.append(Path("../runs") / dataset / f"{layout}{variant}")

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in ["..", ".", "lin", "knn", "recall"]
    }
    return depdict


def aggregate_path(path, outfile=None):
    import h5py
    import numpy as np
    from scipy import sparse

    depd = deps(path)

    with h5py.File(outfile, "x", libver="latest") as f5:
        for k, v in depd.items():
            # same iteration scheme as in `deps()` above so that the order
            # between the zipfile and the parameters match.
            #
            # It's a bit hacky in that the iteration here actually
            # depends on the order of keys in the `depd`.  We first
            # need to create the dataset before we can set the
            # attributes.  Since the dictionary preserves insertion
            # order, this works.  But I don't really like this
            # solution.
            for (dataset, layout, variant), zipf in zip(
                itertools.product(DATASETS, LAYOUTS, VARIANTS), v
            ):
                if k == ".":
                    # read the loss from the run lightning_logs/metrics.csv
                    embedding = np.load(zipf)["embedding"]
                    name = f"{layout}{variant.replace(",", "/")}"
                    name += "/default" if variant == "" else ""
                    f5.create_dataset(f"{dataset}/{name}", data=embedding)
                elif k == "..":
                    if f"{dataset}/labels" not in f5:
                        labels = np.load(zipf)["labels"]
                        f5.create_dataset(f"{dataset}/labels", data=labels)

                    if f"{dataset}/edges" not in f5:
                        A = sparse.load_npz(zipf).tocoo()
                        for attr in ["row", "col"]:  # , "data"]:
                            f5.create_dataset(
                                f"{dataset}/edges/{attr}",
                                data=getattr(A, attr),
                            )

                else:
                    acctxt = (zipfile.Path(zipf) / "score.txt").read_text()
                    score = float(acctxt)
                    f5[f"{dataset}/{layout}"].attrs[k] = score
