import argparse
import importlib
from pathlib import Path


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
        from matplotlib import font_manager

        fonts = font_manager.findSystemFonts(
            [project_root / "media/fonts/ttf"]
        )
        [font_manager.fontManager.addfont(fontpath) for fontpath in fonts]
        with matplotlib.style.context(stylefile):
            mod.plot_path(plotname, outfile, format=args.format)


if __name__ == "__main__":
    main()
