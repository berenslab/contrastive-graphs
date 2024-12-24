import string

DATASETS = ["cora", "computer", "photo", "citeseer", "mnist"]
DATASETS = ["cora", "mnist"]
LAYOUTS = ["drgraph", "tsne"]
# N_EPOCHS = 30


# example plotname = "temperatures"
def deplist(dispatch):
    assert dispatch.name == "layouts"
    return ["../dataframes/all_layouts.h5"]


def plot_path(plotname, outfile, format="pdf"):
    import h5py

    h5file = deplist(plotname)[0]

    with h5py.File(h5file) as h5:
        return plot(h5, outfile=outfile, format=format)


def plot(h5, outfile, format="pdf"):
    from matplotlib import pyplot as plt

    from ..plot import letter_dict

    letters = iter(string.ascii_lowercase)
    fig = plt.figure(figsize=(3, 1.1 * len(h5)))
    figs = fig.subfigures(len(h5))
    for sfig, dataset in zip(figs, h5):
        sfig.suptitle(dataset)
        h5_ds = h5[dataset]
        keys = [k for k in h5_ds if k != "labels"]
        labels = h5_ds["labels"]

        axd = sfig.subplot_mosaic([keys])
        for key, ax in axd.items():
            ax.set_title(key)

            data = h5_ds[key]
            ax.scatter(data[:, 0], data[:, 1], c=labels, rasterized=True)
            ax.set_title(next(letters), **letter_dict())

    fig.savefig(outfile, format=format)
