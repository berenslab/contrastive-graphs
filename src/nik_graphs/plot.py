import argparse
import importlib
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--plotname", required=True, type=Path)
    parser.add_argument("--outfile", type=Path)
    parser.add_argument("--printdeps", action="store_true")
    args = parser.parse_args()

    plotname = args.plotname
    outfile = args.outfile
    # path.mkdir(parents=True, exist_ok=True)

    modulename = plotname.name.split(".")[-1]

    mod = importlib.import_module(
        f".plotting.{modulename}", package="nik_graphs"
    )

    if args.printdeps:
        deps = mod.deplist(plotname)
        stylefile = Path(__file__).parent / "plotting/jnb.mplstyle"
        [print(dep) for dep in deps + [mod.__file__, __file__, stylefile]]
    else:
        mod.plot_path(plotname, outfile)


if __name__ == "__main__":
    main()
