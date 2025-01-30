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
        figsize=(5, 1.8),
        constrained_layout=dict(w_pad=0, h_pad=0.005),
        width_ratios=[0.75, 1, 1.6],
    )

    with plt.rc_context(
        {"lines.markersize": 40**0.5, "scatter.edgecolors": "white"}
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
            shrinkB=5,
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
            shrinkB=3,
        ),
    )
    fig.add_artist(annot)

    kws = dict(ha="right", fontsize=10)
    t = mpl.transforms.blended_transform_factory(
        fig.transSubfigure, axd["t"].transAxes
    )
    x_txt = 0.25
    fig.text(
        x_txt - 0.01,
        0.49,
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
        x_txt - 0.02,
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
        1,
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
    t = mpl.transforms.blended_transform_factory(
        ax.figure.transSubfigure, ax.transAxes
    )
    ax.text(
        0,
        0.55,
        r"$G = (\mathcal V, \mathcal E)$",
        usetex=usetex,
        transform=t,
        fontsize=plt.rcParams["axes.titlesize"],
    ).set_in_layout(False)

    for name, idx in dict(i=5, j=4).items():
        t = ax.figure.dpi_scale_trans + mpl.transforms.ScaledTranslation(
            *pts[idx], ax.transData
        )
        ax.text(
            -2 / 72,
            0,
            f"${name}$",
            usetex=usetex,
            transform=t,
            ha="right",
            va="bottom",
        )

    ax.set_axis_off()
    ax.set_aspect(1)
    ax.scatter(*pts.T, c="xkcd:dark grey")

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
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(translate_plotname("cne,temp=0.05"))
    ax.margins(0.01)

    lcolor = "xkcd:dark grey"
    largs = dict(color=lcolor, lw=plt.rcParams["axes.linewidth"])
    # Plot the surface
    circle = mpl.patches.Circle(
        (0, 0), 1, facecolor="none", edgecolor=lcolor, lw=largs["lw"]
    )
    ax.update_datalim(circle.get_extents())
    ax.add_artist(circle)
    # ax.plot_wireframe(
    #     xs, ys, zs, color="xkcd:slate grey", alpha=0.3, zorder=1, lw=0.1
    # )
    x1, x2 = project_sphere_points(
        *np.linspace([-np.pi / 2, 0], [np.pi / 2, 0], endpoint=False).T
    )
    ax.plot(x2, x1, ls="dashed", **largs)
    ax.plot(x2, -x1, ls="solid", **largs)

    data = pts * 0.5
    data[:, 0] += 0.3
    data[:, 1] *= -1
    lon, lat = data.T
    x1, x2 = project_sphere_points(lon, lat)
    ax.scatter(x1, x2, c="xkcd:dark grey", alpha=1, zorder=8)

    row, col = np.triu(A).nonzero()
    edges = np.hstack((data[row], data[col])).reshape(len(row), 2, 2)

    for edge in edges:
        pta, ptb = edge
        elon, elat = np.linspace(*edge, num=10).T
        x1, x2 = project_sphere_points(elon, elat)

        ax.plot(x1, x2, color="xkcd:slate grey", zorder=5, alpha=1)


def project_sphere_points(
    lon_rad, lat_rad, radius=1.0, perspective_distance=15.0, view_angle=-80
):
    """
    Project spherical coordinates to 3D and then to 2D with perspective.

    Args:
        lon, lat: Arrays of longitude and latitude in radians
        radius: Sphere radius
        perspective_distance: Distance of viewer from sphere center
        view_angle: Rotation angle around vertical axis in degrees

    Returns:
        x, y: Arrays of 2D projected coordinates
    """
    # Convert to radians
    # lon_rad = np.radians(lon)
    # lat_rad = np.radians(lat)
    theta = np.radians(view_angle)

    # Convert to 3D cartesian coordinates
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    # Rotate around y-axis
    x_rot = x * np.cos(theta) + z * np.sin(theta)
    z_rot = -x * np.sin(theta) + z * np.cos(theta)

    # Apply perspective projection
    scale = perspective_distance / (perspective_distance - z_rot)
    x_proj = scale * x_rot
    y_proj = scale * y

    return x_proj, y_proj


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
