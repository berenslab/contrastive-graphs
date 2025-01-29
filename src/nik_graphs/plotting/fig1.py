import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from ..plot import add_letters, translate_plotname

usetex = False


def deplist(plotname=None):
    return []


def plot_path(plotname, outfile, format="pdf"):

    fig = plot()
    fig.savefig(outfile, format="pdf", metadata=dict(CreationDate=None))


def plot():
    import matplotlib as mpl
    import numpy as np

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
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(4, 1.8),
        per_subplot_kw=dict(c=dict(projection="3d")),
        constrained_layout=dict(w_pad=0, h_pad=0),
    )

    # pts1 = rng.normal(pts, 0.1)
    with plt.rc_context({"lines.markersize": 15**0.5}):
        plot_graph(axd["g"], pts, A)
        plot_tsne(axd["t"], pts, A, rng.integers(2**31 - 1))
        plot_cne(axd["c"], pts, A)

    with plt.rc_context({"font.size": 14}):
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
    x_txt = 0.32
    fig.text(x_txt - 0.01, 0.5, "graph layout", transform=t, **kws)
    t = mpl.transforms.blended_transform_factory(
        fig.transSubfigure, axd["c"].transAxes
    )
    fig.text(x_txt + 0.0, 0.5, "node embedding", transform=t, **kws)

    t = mpl.transforms.blended_transform_factory(
        axd["t"].transAxes, axd["i"].transAxes
    )
    fig.text(
        0.85,
        1,
        r"$\mathbb{S}^{127}$",
        ha="left",
        va="top",
        usetex=usetex,
        transform=t,
        fontsize=14,
        # zdir="y",
    )

    add_letters(axd[ltr] for ltr in "gtcki")
    return fig


def plot_graph(ax, pts, A):
    # ax.text(
    #     0.25,
    #     1,
    #     "graph",
    #     va="top",
    #     transform=ax.transAxes,
    #     fontsize=plt.rcParams["axes.titlesize"],
    # )
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
        # early_exaggeration_iter=0,
        random_state=random_state,
    )
    rot, _scale = linalg.orthogonal_procrustes(Y, pts)
    data = Y @ rot.round(10)

    ax.tick_params("both", length=0)
    ax.set(xticks=[], yticks=[])  # , xlabel="$t$-SNE 1", ylabel="$t$-SNE 2")
    # [ax.spines[m].set_visible(True) for m in ["right", "top"]]
    [ax.spines[m].set_visible(False) for m in ["left", "bottom"]]
    kwargs = dict(
        xycoords=ax.transAxes,
        clip_on=False,
        arrowprops=dict(
            arrowstyle="<|-",
            color="xkcd:dark grey",
            # connectionstyle="arc3,rad=0.3",
            shrinkA=0,
            shrinkB=0,
            lw=plt.rcParams["axes.linewidth"],
        ),
    )
    ax.annotate("", (0, 0), (0, 1), **kwargs)
    ax.annotate("", (0, 0), (1, 0), **kwargs)

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

    # theta = np.linspace(0, 2 * np.pi, 100)
    # z = np.zeros(100)
    # x = np.sin(theta)
    # y = np.cos(theta)
    # ax.plot(x, y, z, color="black", alpha=0.75)
    # ax.plot(z, x, y, color="black", alpha=0.75)

    data = pts * 1.1
    lon, lat = data.T
    # rho = 1
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
        r"$\ell_{ij} = "
        r"-\log\frac{(1 + d_{ij}^2)^{-1}}{\sum_{kl}(1 + d_{kl}^2)^{-1}}$"
    )
    ax.text(
        0.5,
        0.5,
        loss,
        usetex=usetex,
        transform=ax.transAxes,
        ha="center",
        va="center",
    )


def plot_infonce(ax):
    ax.set_axis_off()
    ax.set_title("InfoNCE")

    loss = (
        r"$\ell_{ij} = "
        r"-\log\frac{\exp(y_i y_j / \tau)}{\sum_k\exp(y_i y_j / \tau)}$"
    )

    ax.text(
        0.5,
        0.5,
        loss,
        usetex=usetex,
        transform=ax.transAxes,
        ha="center",
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
        # rasterized=True,
    )
    return lines
