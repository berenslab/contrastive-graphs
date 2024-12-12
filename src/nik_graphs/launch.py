import argparse
import importlib
from pathlib import Path

# import nik_graphs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--outfile", required=True, type=Path)
    args = parser.parse_args()

    path = args.path
    outfile = args.outfile

    modulename = path.name.split(",")[0]

    mod = importlib.import_module(
        f".modules.{modulename}", package="nik_graphs"
    )
    mod.run_path(path, outfile)


if __name__ == "__main__":
    main()
