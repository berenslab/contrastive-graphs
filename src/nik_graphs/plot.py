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
        ax.set_title(letter, **ld)


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


def translate_plotname(x):
    match x:
        case "lin":
            s = "linear"
        case "knn":
            s = "$k$NN"
        case "recall":
            s = "recall"
        case _:
            raise ValueError(f"Unknown value {x!r} for translating")
    return s


if __name__ == "__main__":
    main()
