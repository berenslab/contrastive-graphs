import matplotlib as mpl
import numpy as np


def deplist(plotname=None):
    return []


def plot_path(plotname, outfile, format="pdf"):

    fig = plot()
    fig.savefig(outfile, format="pdf", metadata=dict(CreationDate=None))


def plot():
    import numpy as np
    from matplotlib import pyplot as plt

    rng = np.random.default_rng(101)

    pts = np.array(
        [[-2, -1.5], [-0.8, -1], [0, 0], [0.9, 0.5], [0.6, 1.2], [-0.1, 0.75]]
    )

    D = ((pts - pts[:, None]) ** 2).sum(-1)
    A = (D < 2) - np.diag(np.ones(len(pts)))
    row, col = A.nonzero()

    mosaic = """
    .tk
    gtk
    gci
    .ci
    """
    # mosaic = [["graph", "tsne", "kl"], ["graph", "cne", "infonce"]]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(6.75, 2))

    # pts1 = rng.normal(pts, 0.1)
    with plt.rc_context({"lines.markersize": 15**0.5}):
        plot_graph(axd["g"], pts, A)
        plot_tsne(axd["t"], pts, A, rng.integers(2**31 - 1))

    return fig


def plot_graph(ax, pts, A):
    ax.set_axis_off()
    ax.set_aspect(1)
    ax.scatter(*pts.T, c="xkcd:dark grey", s=15)

    ax.add_collection(get_edgelines(pts, A))


def plot_tsne(ax, pts, A, random_state=5):
    from nik_graphs.modules.tsne import tsne
    from scipy import linalg

    Y = tsne(
        A, negative_gradient_method="bh", theta=0, random_state=random_state
    )
    rot, _scale = linalg.orthogonal_procrustes(Y, pts)
    data = Y @ rot.round(10)

    ax.scatter(*data.T)
    ax.set_aspect(1)
    ax.add_collection(get_edgelines(data, A))
    ax.set_title("$t$-SNE")


def plot_cne(ax): ...


def plot_kl(ax): ...


def plot_infonce(ax): ...


def get_edgelines(pts, A):
    row, col = np.triu(A).nonzero()
    pts_e = np.hstack((pts[row], pts[col])).reshape(len(row), 2, 2)
    lines = mpl.collections.LineCollection(
        pts_e,
        alpha=0.8,
        color="xkcd:slate grey",
        antialiaseds=True,
        zorder=0.9,
        rasterized=True,
    )
    return lines
