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

    if args.printdeps:
        deps = mod.deplist(dispatch)
        [print(dep) for dep in deps + [mod.__file__, __file__]]
    else:
        return mod.aggregate_path(dispatch, outfile)


if __name__ == "__main__":
    main()
