import itertools
import zipfile
from pathlib import Path

DATASETS = ["cora", "computer", "photo", "citeseer", "mnist"]
LAYOUTS = ["fa2", "tfdp", "drgraph", "tsne"]


# example plotname = "temperatures"
def deplist(dispatch):
    depvals = deps(dispatch).values()
    return [f for lis in depvals for f in lis]


def deps(dispatch):
    # dataset, algo, _name = dispatch.name.split(".")
    assert str(dispatch) == "all_layouts"

    dataset = "mnist"

    paths = []
    for dataset, layout in itertools.product(DATASETS, LAYOUTS):
        match layout:
            case "cne":
                x = "cne,dim=2"
            case "fa2":
                x = "nx,layout=forceatlas2"
            case "fdp":
                x = "nx,layout=spring"
            case _:
                x = layout
        paths.append(Path("../runs") / dataset / x)

    depdict = {
        k: [p / k / "1.zip" for p in paths]
        for k in ["..", ".", "lin", "knn", "recall"]
    }
    return depdict


def aggregate_path(path, outfile=None):
    import h5py
    import numpy as np

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
            for (dataset, layout), zipf in zip(
                itertools.product(DATASETS, LAYOUTS), v
            ):
                if k == ".":
                    # read the loss from the run lightning_logs/metrics.csv
                    embedding = np.load(zipf)["embedding"]
                    f5.create_dataset(f"{dataset}/{layout}", data=embedding)
                elif k == "..":
                    labels = np.load(zipf)["labels"]
                    if f"{dataset}/labels" not in f5:
                        f5.create_dataset(f"{dataset}/labels", data=labels)
                else:
                    acctxt = (zipfile.Path(zipf) / "score.txt").read_text()
                    score = float(acctxt)
                    f5[f"{dataset}/{layout}"].attrs[k] = score
