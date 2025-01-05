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
        default="tex",
        help="parameter that determines the output file type.",
    )
    args = parser.parse_args()

    dispatch = args.dispatch
    outfile = args.outfile
    # path.mkdir(parents=True, exist_ok=True)

    modulename = dispatch.name.split(".")[-1]

    mod = importlib.import_module(
        f".tables.{modulename}", package="nik_graphs"
    )

    if args.printdeps:
        deps = mod.deplist(dispatch, format=args.format)
        [print(dep) for dep in deps + [mod.__file__, __file__]]
    else:
        return mod.format_table(dispatch, outfile, format=args.format)


if __name__ == "__main__":
    main()
