# example plotname = "mnist.tsne.summary". The middle part must not
# contain dots as part of the arguments.
def deplist(plotname):
    return list(deps(plotname).values())


def deps(plotname):
    from pathlib import Path

    dataset, algo, _name = plotname.name.split(".")
    assert _name == "summary"

    path = Path("../runs") / dataset / algo
    depdict = {k: path / k / "1.zip" for k in ["lin", "knn", "recall"]}
    depdict["embedding"] = path / "1.zip"
    depdict["data"] = path.parent / "1.zip"
    return depdict


def plot_path(plotname, outfile, format="pdf"):
    import zipfile

    import numpy as np
    from scipy import sparse

    files = deps(plotname)

    embedding = np.load(files["embedding"])["embedding"]
    if embedding.shape[1] > 2:
        raise RuntimeError(
            f"Need a 2D embedding, but given array is {embedding.shape[1]}D"
        )

    labels = np.load(files["data"])["labels"]
    A = sparse.load_npz(files["data"])
    accd = dict()
    for k in ["lin", "knn", "recall"]:
        with zipfile.ZipFile(files[k]) as zf:
            with zf.open("score.txt") as f:
                acc = float(f.read())
        accd[k] = acc

    return plot(embedding, labels, accd, A=A, outfile=outfile, format=format)


def plot(embedding, labels, accd, A=None, outfile=None, format="pdf"):
    import matpotlib as mpl
    import numpy as np
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.scatter(*embedding.T, c=labels, alpha=0.6, rasterized=True)
    ax.set_aspect(1)

    txt = "\n".join(f"{k} = {v:.1%}" for k, v in accd.items())
    ax.text(
        1,
        1,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        ma="right",
        family="monospace",
    )

    if A is not None:
        A = A.tocoo()
        pts = np.hstack((embedding[A.row], embedding[A.col])).reshape(
            len(A.row), 2, 2
        )
        lines = mpl.collections.LineCollection(
            pts,
            alpha=0.05,
            color="xkcd:dark grey",
            antialiaseds=True,
            zorder=0.9,
            rasterized=True,
        )
        ax.add_collection(lines)

    fig.savefig(outfile, format=format)
