import argparse
import importlib
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatch", required=True, type=Path)
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

    dispatch = args.dispatch
    outfile = args.outfile
    # path.mkdir(parents=True, exist_ok=True)

    modulename = dispatch.name.split(".")[-1]

    mod = importlib.import_module(
        f".aggregate.{modulename}", package="nik_graphs"
    )

    project_root = Path(__file__).parent.parent.parent
    stylefile = Path(__file__).parent / "plotting/jnb.mplstyle"
    if args.printdeps:
        deps = mod.deplist(dispatch)
        [print(dep) for dep in deps + [mod.__file__, __file__, stylefile]]
    else:
        return mod.aggregate_path(plotname, outfile)


if __name__ == "__main__":
    main()
