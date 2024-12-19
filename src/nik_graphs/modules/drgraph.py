import inspect
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"

PROJROOT = Path(__file__).parent.parent.parent.parent
BIN = PROJROOT / "bin/drgraph"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n{BIN}\n")

    # this is done so we can bootstrap the executable.  If we call
    # this via redo for the first time, then we do not yet know the
    # dependencies, hence we can run into the case that this function
    # is executing before the dependencies have been tracked (as we do
    # the recording only after running this).
    if not BIN.exists():
        subprocess.run(["redo-ifchange", str(BIN)], check=True)

    with zipfile.ZipFile(zipf) as zf:
        zf.extract("drgraph.txt", path=path)

    A_file = path / "drgraph.txt"

    name, kwargs = path_to_kwargs(path)
    assert name == "drgraph"

    Y = drgraph(A_file, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def drgraph(
    A_file,
    n_epochs=100,
    # initialization="spectral",
    # random_state=505**3,
    **kwargs,
):

    # need to create a system call kind of like:

    # ./bin/drgraph
    # -input ../DRGraph/data/block_2000.txt
    # -output /tmp/block_2000.txt
    # -neg 5
    # -samples 400
    # -gamma 0.1
    # -mode 1
    # -A 2
    # -B 1
    kwargs1 = dict(neg=5, samples=400, gamma=0.1, mode=1, A=2, B=1)
    kwargs1.update(kwargs)

    args = ["-input", f"{A_file}"]
    for k, v in kwargs1.items():
        args += [f"-{k}", f"{v}"]

    with tempfile.NamedTemporaryFile(dir=A_file.parent) as tempf:
        args += ["-output", tempf.name]

        subprocess.run([BIN] + args, check=True, capture_output=True)
        # skip the first row, only contains info about the shape of
        # the array.
        Y = np.loadtxt(tempf.name, skiprows=1)

    return Y
