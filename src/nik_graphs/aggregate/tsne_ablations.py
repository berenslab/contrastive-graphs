import itertools
import zipfile
from pathlib import Path

DATASETS = [
    "computer",
    "photo",
]
LAYOUTS = ["tsne"]
VARIANTD = dict(
    default="", no_row_norm=",row_norm=0", random_init=",initialization=random"
)


# example plotname = "temperatures"
def deplist(dispatch):
    depvals = deps(dispatch).values()
    return list(set([f for lis in depvals for f in lis]))


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "tsne_ablations"

    paths = []
    for dataset, layout, variant in itertools.product(
        DATASETS, LAYOUTS, VARIANTD.values()
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
            for (dataset, layout, variant_name), zipf in zip(
                itertools.product(DATASETS, LAYOUTS, VARIANTD.keys()), v
            ):
                if k == ".":
                    # read the loss from the run lightning_logs/metrics.csv
                    embedding = np.load(zipf)["embedding"]
                    f5.create_dataset(
                        f"{dataset}/{layout}/{variant_name}", data=embedding
                    )
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
                    f5[f"{dataset}/{layout}/{variant_name}"].attrs[k] = score
