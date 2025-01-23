import argparse
import functools
import importlib
import itertools
import string
from pathlib import Path


def letter_dict():
    return dict(
        loc="left",
        horizontalalignment="right",
        fontsize=10,
        weight="bold",
        family="Roboto",
    )


def letter_iterator():
    # gives 27404 letter combinations from a, b, ..., zzzz
    return (
        functools.reduce(lambda a, b: a + b, x, "")
        for i in range(1, 5)
        for x in itertools.combinations_with_replacement(
            string.ascii_lowercase, i
        )
    )


def add_letters(axs):
    ld = letter_dict()
    for letter, ax in zip(letter_iterator(), axs):
        n_newlines = len(ax.get_title().split("\n")) - 1
        ax.set_title(letter + "\n" * n_newlines, **ld)


def add_scalebars(ax, **kwargs):
    from ._scalebars import add_scalebar_frac

    return add_scalebar_frac(ax, **kwargs)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plotname", required=True, type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--printdeps", action="store_true")
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        help="parameter that determines the output file type, will "
        "be passed to the call to `fig.savefig()`.",
    )
    args = parser.parse_args()

    plotname = args.plotname
    outfile = args.outfile
    # path.mkdir(parents=True, exist_ok=True)

    modulename = plotname.name.split(".")[-1]

    mod = importlib.import_module(
        f".plotting.{modulename}", package="nik_graphs"
    )

    project_root = Path(__file__).parent.parent.parent
    stylefile = Path(__file__).parent / "plotting/jnb.mplstyle"
    if args.printdeps:
        deps = mod.deplist(plotname)
        [print(dep) for dep in deps + [mod.__file__, __file__, stylefile]]

        fontfiles = (project_root / "media/fonts/ttf").glob("*")
        [print(f / "all-fonts") for f in fontfiles]

    else:
        # set up fonts
        import matplotlib
        import matplotlib.style
        from matplotlib import font_manager

        fonts = [
            p / f
            for p in (project_root / "media/fonts/ttf").glob("*")
            for f in (p / "all-fonts").read_text().split("\n")
            if f != ""
        ]
        [font_manager.fontManager.addfont(fontpath) for fontpath in fonts]
        with matplotlib.style.context(stylefile):
            mod.plot_path(plotname, outfile, format=args.format)


def translate_plotname(x, _return="error"):
    dataset_capitalize = ["cora", "citeseer", "pubmed", "computer", "photo"]
    dataset_allcaps = ["mag", "sbm"]
    dataset_mapping = dict(arxiv="arXiv", mnist="MNIST $k$NN")
    match x:
        case "lin":
            s = "linear"
        case "knn":
            s = "$k$NN"
        case "recall":
            s = "neighbor recall"
        case "tsne":
            s = "graph $t$-SNE"
        case "tfdp":
            s = "$t$-FDP"
        case "fa2":
            s = "ForceAtlas2"
        case "sgtsnepi":
            s = "SGtSNEpi"
        case "drgraph":
            s = "DRGraph"
        case str(x) if x.startswith("spectral"):
            s = "Laplacian E."
        case "cne,temp=0.05":
            s = "graph CNE"
        case "cne":
            s = "CNE, τ=0.5"
        case "cne,loss=infonce-temp":
            s = "CNEτ"
        case "deepwalk":
            s = "DeepWalk"
        case "node2vec":
            s = x
        case str(x) if x in dataset_capitalize:
            s = x.title()
        case str(x) if x in dataset_allcaps:
            s = x.upper()
        case str(x) if x in dataset_mapping:
            s = dataset_mapping[x]
        case _:
            if _return == "identity":
                s = x
            elif _return == "error":
                raise ValueError(f"Unknown value {x!r} for translating")
            else:
                raise ValueError(
                    f"Unknown value {x!r} and {_return=!r} (for translating"
                )
    if x in ["lin", "knn"]:
        s += " accuracy"
    return s


def translate_acc_short(x):
    match x:
        case "knn":
            s = "$k$NN acc."
        case "lin":
            s = "lin. acc."
        case "recall":
            s = x
        case _:
            raise ValueError(f"Unknown value {x!r} (for translating")

    return s


def name2color(x, _return="raise_error"):
    match x:
        case "tsne" | "cne,temp=0.05" | "cne,loss=infonce-temp":
            c = "xkcd:blue"
        case "sgtsnepi":
            c = "xkcd:orange"
        case "tfdp":
            c = "xkcd:deep lavender"
        case "fa2":
            c = "xkcd:pink red"
        case "drgraph":
            c = "xkcd:kelly green"
        case str(x) if x.startswith("spectral"):
            c = "xkcd:brown"
        case "deepwalk":
            c = "xkcd:salmon"
        case "node2vec":
            c = "xkcd:light moss green"
        case _:
            if _return is None:
                c = None
            else:
                raise ValueError(f"Uknown name to map to a color {x!r}")
    return c


if __name__ == "__main__":
    main()
