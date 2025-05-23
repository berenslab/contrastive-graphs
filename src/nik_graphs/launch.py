import argparse
import importlib
import inspect
import time
import zipfile
from pathlib import Path

# import nik_graphs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--outfile", required=True, type=Path)
    args = parser.parse_args()

    path = args.path
    outfile = args.outfile
    path.mkdir(parents=True, exist_ok=True)

    modulename = path.name.split(",")[0]

    mod = importlib.import_module(
        f".modules.{modulename}", package="nik_graphs"
    )

    # truncates the file, we do want that here (basically a dirty
    # reset)
    (path / "files.dep").write_text("")

    t0 = time.perf_counter()
    mod.run_path(path, outfile)
    t1 = time.perf_counter()

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("elapsed_secs.txt", "w") as f:
            f.write(f"{t1 - t0}\n".encode())

    deptxt = f"{inspect.getfile(mod)}\n{__file__}\n"
    with open(path / "files.dep", "a") as f:
        f.write(deptxt)


if __name__ == "__main__":
    main()
