from pathlib import Path

import nik_graphs


def run_path(p, outfile):
    p = Path(p)
    fpath = Path(__file__).resolve()

    print("Hello from ", fpath, f":\t{p = }")
