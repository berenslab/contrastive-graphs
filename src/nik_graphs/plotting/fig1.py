from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from ..plot import translate_plotname

usetex = True
graph_color = "black"
graph_edge_color = graph_color
axes_edge_color = "xkcd:medium grey"

ij_dict = dict(i=5, j=4)
attraction_color = "crimson"
repulsion_color = "xkcd:ocean blue"
_attr = mpl.colors.to_hex(attraction_color)[1:]
_repl = mpl.colors.to_hex(repulsion_color)[1:]
plt.rcParams.update(
    {
        f"{x}.preamble": "\n".join(
            [
                r"\usepackage{xcolor}",
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
                rf"\definecolor{{attr}}{{HTML}}{{{_attr}}}",
                rf"\definecolor{{repl}}{{HTML}}{{{_repl}}}",
            ]
        )
        for x in ["text.latex", "pgf"]
    }
)


def deplist(plotname=None):
    return ["../bin/texlive"] if usetex else []


def plot_path(plotname, outfile, format="pdf"):

    fig = plot()
    kws = dict(metadata=dict(CreationDate=None)) if format == "pdf" else dict()
    fig.savefig(outfile, format=format, **kws)


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
    .ci
    gci
    gtk
    .tk
    """
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(5.5, 1.8),
        constrained_layout=dict(w_pad=0, h_pad=0),
    )

    with plt.rc_context(
        {"lines.markersize": 40**0.5, "scatter.edgecolors": "white"}
    ):
        plot_graph(axd["g"], pts, A)
        plot_tsne(axd["t"], pts, A, rng.integers(2**31 - 1))
        plot_cne(axd["c"], pts, A)

    with plt.rc_context({"font.size": 10}):
        plot_kl(axd["k"])
        plot_infonce(axd["i"])

    annot = plt.Annotation(
        "",
        (0, 0.6),
        (1, 0.85),
        textcoords=axd["g"].transAxes,
        xycoords=axd["c"].transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            color="xkcd:dark grey",
            connectionstyle="arc3,rad=-0.225",
            shrinkB=5,
        ),
    )
    fig.add_artist(annot)
    annot = plt.Annotation(
        "",
        (0, 0.5),
        (0.95, 0.53),
        textcoords=axd["g"].transAxes,
        xycoords=axd["t"].transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            color="xkcd:dark grey",
            connectionstyle="arc3,rad=0.225",
            shrinkA=0,
            shrinkB=3,
        ),
    )
    fig.add_artist(annot)

    kws = dict(ha="right", fontsize=8)
    t = mpl.transforms.blended_transform_factory(
        fig.transSubfigure, axd["c"].transAxes
    )
    x_txt = 0.275
    fig.text(
        x_txt - 0.01,
        0.6,
        "node\nembedding",
        ma="right",
        va="bottom",
        transform=t,
        **kws,
    )
    t = mpl.transforms.blended_transform_factory(
        fig.transSubfigure, axd["t"].transAxes
    )
    fig.text(
        x_txt - 0,
        0.575,
        "graph\nlayout",
        ma="right",
        va="top",
        transform=t,
        **kws,
    )

    t = mpl.transforms.blended_transform_factory(
        axd["t"].transAxes, axd["i"].transAxes
    )
    fig.text(
        0.95,
        1,
        r"$S^{127}$",
        ha="left",
        va="top",
        usetex=usetex,
        transform=t,
        fontsize=10,
    )

    return fig


def plot_graph(ax, pts, A):
    t = mpl.transforms.blended_transform_factory(
        ax.figure.transSubfigure, ax.transAxes
    )
    ax.text(
        0.0125,
        0.55,
        r"$G = (\mathcal V, \mathcal E)$",
        usetex=usetex,
        transform=t,
        fontsize=plt.rcParams["axes.titlesize"],
    ).set_in_layout(False)

    for name, idx in ij_dict.items():
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
            color=attraction_color,
        )

    ax.set_axis_off()
    ax.set_aspect(1)
    colors = [
        attraction_color if x in ij_dict.values() else graph_color
        for x in range(len(pts))
    ]
    ax.scatter(*pts.T, c=colors)

    ax.add_collection(get_edgelines(pts, A))


def plot_tsne(root_ax, pts, A, random_state=5):
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

    root_ax.set_aspect(1)
    root_ax.set_axis_off()
    w = 0.85
    ax = root_ax.inset_axes([(1 - w) / 2, 0.075, w, 0.875])
    ax.tick_params("both", length=0)
    ax.set(xticks=[], yticks=[])
    [
        (s.set_visible(True), s.set_color(axes_edge_color))
        for s in ax.spines.values()
    ]

    ax.margins(0.1)
    colors = [
        attraction_color if x in ij_dict.values() else graph_color
        for x in range(len(pts))
    ]
    ax.scatter(*data.T, c=colors)
    ax.set_aspect(1)
    ax.add_collection(get_edgelines(data, A))
    ax.text(
        1.025,
        1,
        r"$\mathbb{R}^2$",
        transform=ax.transAxes,
        usetex=usetex,
        ha="left",
        va="top",
        fontsize=10,
    )


def plot_cne(ax, pts, A):
    ax.set_aspect("equal")
    ax.tick_params("both", length=0)
    [
        axis.set_major_formatter(mpl.ticker.NullFormatter())
        for axis in [ax.xaxis, ax.yaxis]
    ]
    [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("graph NE")
    ax.margins(0.01)

    largs = dict(color=axes_edge_color, lw=plt.rcParams["axes.linewidth"])
    # Plot the surface
    circle = mpl.patches.Circle(
        (0, 0), 1, facecolor="none", edgecolor=largs["color"], lw=largs["lw"]
    )
    ax.update_datalim(circle.get_extents())
    ax.add_artist(circle)
    x1, x2 = project_sphere_points(
        *np.linspace([-np.pi / 2, 0], [np.pi / 2, 0], endpoint=False).T
    )
    ax.plot(x2, x1, ls="dashed", **largs)
    ax.plot(x2, -x1, ls="solid", **largs)

    data = pts * 0.6
    data[:, 1] += np.pi / 32
    lon, lat = data.T
    x1, x2 = project_sphere_points(lat, lon)
    for i, (lo, la, c1, c2) in enumerate(zip(lat, lon, x1, x2)):
        major, minor, angle = analyze_projected_circle(lo, la, 5)
        color = attraction_color if i in ij_dict.values() else graph_color
        ellipse = mpl.patches.Ellipse(
            (c1, c2),
            major,
            minor,
            angle=angle,
            facecolor=color,
            edgecolor="white",
            zorder=3,
        )
        ax.add_patch(ellipse)

    row, col = np.triu(A).nonzero()
    edges = np.hstack((data[row], data[col])).reshape(len(row), 2, 2)

    for edge, i, j in zip(edges, row, col):
        pta, ptb = edge
        elon, elat = np.linspace(*edge, num=10).T
        x1, x2 = project_sphere_points(elat, elon)

        c = (
            graph_edge_color
            if not (j == ij_dict["i"] and i == ij_dict["j"])
            else attraction_color
        )
        ax.plot(x1, x2, color=c, alpha=1)


def project_sphere_points(
    lon_rad, lat_rad, radius=1.0, perspective_distance=15.0, view_angle=80
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


def analyze_projected_circle(
    center_lon_rad,
    center_lat_rad,
    circle_radius,
    num_points=100,
    **projection_kwargs,
):
    """
    Analyze how a circle on the sphere projects to an ellipse in 2D.

    Args:
        center_lon, center_lat: Circle center coordinates in degrees
        circle_radius: Angular radius of the circle in degrees
        num_points: Number of points to sample on the circle
        **projection_kwargs: Arguments passed to project_sphere_points

    Returns:
        major_axis: Length of major axis
        minor_axis: Length of minor axis
        angle: Rotation angle of the ellipse in degrees
    """
    # Generate points around the circle
    t = np.linspace(0, 2 * np.pi, num_points)

    # Convert circle_radius to radians
    r = np.radians(circle_radius)

    # Generate circle points using spherical trigonometry
    lat = np.arcsin(
        np.sin(center_lat_rad) * np.cos(r)
        + np.cos(center_lat_rad) * np.sin(r) * np.cos(t)
    )

    lon = center_lon_rad + np.arctan2(
        np.sin(r) * np.sin(t),
        np.cos(center_lat_rad) * np.cos(r)
        - np.sin(center_lat_rad) * np.sin(r) * np.cos(t),
    )

    # Project points
    x, y = project_sphere_points(lon, lat, **projection_kwargs)

    # Fit ellipse using covariance matrix
    points = np.column_stack([x - np.mean(x), y - np.mean(y)])
    cov = points.T @ points / (len(points) - 1)
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # Calculate ellipse parameters
    major_axis = 2 * np.sqrt(eigenvals[1])
    minor_axis = 2 * np.sqrt(eigenvals[0])
    angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))

    return major_axis, minor_axis, angle


def plot_kl(ax):
    ax.set_axis_off()

    ax.text(
        0.5,
        1,
        # "Kullback–Leibler divergence",
        "KL divergence",
        va="top",
        ha="center",
        # fontsize=plt.rcParams["axes.titlesize"],
        fontsize=10,
    )

    loss = (
        r"$\displaystyle\ell_{ij} = "
        r"-\log\frac{\color{attr}(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}"
        r"{\color{repl}\sum_{kl}(1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$"
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
    ax.text(
        0.5,
        1,
        "InfoNCE",
        va="top",
        ha="center",
        fontsize=10,  # fontsize=plt.rcParams["axes.titlesize"]
    )

    loss = (
        r"$\displaystyle\ell_{ij} = "
        r"-\log\frac{\color{attr}\exp(\mathbf y_i^\top \mathbf y_j / \tau)}"
        r"{\color{repl}\sum_k\exp(\mathbf y_i^\top \mathbf y_k / \tau)}$"
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

    colors = [
        (
            graph_edge_color
            if not (j == ij_dict["i"] and i == ij_dict["j"])
            else attraction_color
        )
        for i, j in zip(row, col)
    ]
    lines = mpl.collections.LineCollection(
        pts_e,
        color=colors,
        antialiaseds=True,
        zorder=0.9,
    )
    return lines
