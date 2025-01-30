from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from ..plot import letter_dict, translate_plotname

usetex = True

plt.rcParams.update(
    {
        "text.latex.preamble": "\n".join(
            [r"\usepackage{amsmath}", r"\usepackage{amssymb}"]
        )
    }
)


def deplist(plotname=None):
    return ["../bin/texlive"]


def plot_path(plotname, outfile, format="pdf"):

    fig = plot()
    fig.savefig(outfile, format="pdf", metadata=dict(CreationDate=None))


def plot():
    import os

    import matplotlib as mpl
    import numpy as np

    # bin/tex/texlive/2025/bin/x86_64-linux/pdflatex
    project_root = Path(__file__).parent.parent.parent.parent
    os.environ["PATH"] = (
        f"{project_root / 'bin/tex/texlive/2025/bin/x86_64-linux'}:"
        + os.environ["PATH"]
    )

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
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(4.3, 1.8),
        per_subplot_kw=dict(c=dict(projection="3d")),
        constrained_layout=dict(w_pad=0, h_pad=0),
        width_ratios=[0.75, 1, 1.6],
    )

    with plt.rc_context(
        {"lines.markersize": 25**0.5, "scatter.edgecolors": "white"}
    ):
        plot_graph(axd["g"], pts, A)
        plot_tsne(axd["t"], pts, A, rng.integers(2**31 - 1))
        plot_cne(axd["c"], pts, A)

    with plt.rc_context({"font.size": 12}):
        plot_kl(axd["k"])
        plot_infonce(axd["i"])

    annot = plt.Annotation(
        "",
        (0, 0.5),
        (1, 0.85),
        textcoords=axd["g"].transAxes,
        xycoords=axd["t"].transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            color="xkcd:dark grey",
            connectionstyle="arc3,rad=-0.3",
            shrinkB=7.5,
        ),
    )
    fig.add_artist(annot)
    annot = plt.Annotation(
        "",
        (0, 0.5),
        (0.95, 0.5),
        textcoords=axd["g"].transAxes,
        xycoords=axd["c"].transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            color="xkcd:dark grey",
            connectionstyle="arc3,rad=0.3",
            shrinkA=0,
            shrinkB=0,
        ),
    )
    fig.add_artist(annot)

    kws = dict(ha="right", fontsize=10)
    t = mpl.transforms.blended_transform_factory(
        fig.transSubfigure, axd["t"].transAxes
    )
    x_txt = 0.24
    fig.text(
        x_txt - 0.01,
        0.45,
        "graph\nlayout",
        ma="right",
        va="bottom",
        transform=t,
        **kws,
    )
    t = mpl.transforms.blended_transform_factory(
        fig.transSubfigure, axd["c"].transAxes
    )
    fig.text(
        x_txt + 0.0,
        0.6,
        "node\nembedding",
        ma="right",
        va="top",
        transform=t,
        **kws,
    )

    t = mpl.transforms.blended_transform_factory(
        axd["t"].transAxes, axd["i"].transAxes
    )
    fig.text(
        0.85,
        1,
        r"$S^{127}$",
        ha="left",
        va="top",
        usetex=usetex,
        transform=t,
        fontsize=14,
    )

    return fig


def plot_graph(ax, pts, A):
    ax.text(
        0,
        0.6,
        r"$G = (\mathcal V, \mathcal E)$",
        usetex=usetex,
        va="top",
        transform=ax.transAxes,
        fontsize=plt.rcParams["axes.titlesize"],
    )
    ax.set_axis_off()
    ax.set_aspect(1)
    ax.scatter(*pts.T, c="xkcd:dark grey", s=15)

    ax.add_collection(get_edgelines(pts, A))


def plot_tsne(ax, pts, A, random_state=5):
    from scipy import linalg

    from ..modules.tsne import tsne

    Y = tsne(
        A,
        negative_gradient_method="bh",
        theta=0,
        initialization="random",
        n_epochs=100,
        random_state=random_state,
    )
    rot, _scale = linalg.orthogonal_procrustes(Y, pts)
    data = Y @ rot.round(10)

    ax.tick_params("both", length=0)
    ax.set(xticks=[], yticks=[])
    [s.set_visible(True) for s in ax.spines.values()]

    ax.margins(0.1)
    ax.scatter(*data.T, c="xkcd:dark grey")
    ax.set_aspect(1)
    ax.add_collection(get_edgelines(data, A))
    ax.set_title(translate_plotname("tsne"))
    ax.text(
        1.025,
        0.95,
        r"$\mathbb{R}^2$",
        transform=ax.transAxes,
        usetex=usetex,
        ha="left",
        va="top",
        fontsize=14,
    )


def plot_cne(ax, pts, A):
    zdir = "z"

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(translate_plotname("cne,temp=0.05"))
    ax.view_init(azim=45)
    ax.set(zlim=(-1, 1), xlim=(-1, 1), ylim=(-1, 1))

    nsamp = 20
    u = np.linspace(0, 2 * np.pi, nsamp)
    v = np.linspace(0, np.pi, nsamp)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_wireframe(
        xs, ys, zs, color="xkcd:slate grey", alpha=0.3, zorder=1, lw=0.1
    )
    # ax.plot_surface(xs, ys, zs, color="xkcd:light grey", alpha=0.5)

    data = pts * 1.1
    lon, lat = data.T
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    ax.scatter(x, y, z, c="xkcd:dark grey", alpha=1, zdir=zdir, zorder=8)

    row, col = np.triu(A).nonzero()
    edges = np.hstack((data[row], data[col])).reshape(len(row), 2, 2)

    for edge in edges:
        pta, ptb = edge
        elon, elat = np.linspace(*edge, num=10).T
        ex = np.cos(elat) * np.cos(elon)
        ey = np.cos(elat) * np.sin(elon)
        ez = np.sin(elat)

        ax.plot(
            ex, ey, ez, color="xkcd:slate grey", zdir=zdir, zorder=5, alpha=1
        )


def plot_kl(ax):
    ax.set_axis_off()

    ax.set_title("Kullback–Leibler div.")

    loss = (
        r"$\displaystyle\ell_{ij} = "
        r"-\log\frac{(1 + ||\mathbf{y}_i - \mathbf{y}_j||^2)^{-1}}"
        r"{\sum_{kl}(1 + ||\mathbf{y}_k - \mathbf{y}_l||^2)^{-1}}$"
    )
    ax.text(
        0.0,
        0.5,
        loss,
        usetex=usetex,
        transform=ax.transAxes,
        ha="left",
        va="center",
    )


def plot_infonce(ax):
    ax.set_axis_off()
    ax.set_title("InfoNCE")

    loss = (
        r"$\displaystyle\ell_{ij} = "
        r"-\log\frac{\exp(\mathbf y_i^\top \mathbf y_j / \tau)}"
        r"{\sum_k\exp(\mathbf y_i^\top \mathbf y_j / \tau)}$"
    )

    ax.text(
        0,
        0.5,
        loss,
        usetex=usetex,
        transform=ax.transAxes,
        ha="left",
        va="center",
    )


def get_edgelines(pts, A):
    row, col = np.triu(A).nonzero()
    pts_e = np.hstack((pts[row], pts[col])).reshape(len(row), 2, 2)
    lines = mpl.collections.LineCollection(
        pts_e,
        alpha=0.8,
        color="xkcd:slate grey",
        antialiaseds=True,
        zorder=0.9,
    )
    return lines
