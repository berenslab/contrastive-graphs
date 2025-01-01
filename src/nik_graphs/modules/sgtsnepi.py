import inspect
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse

from ..path_utils import path_to_kwargs
from .nx import _get_init

__partition__ = "cpu-galvani"

PROJROOT = Path(__file__).parent.parent.parent.parent
# this is the julia executable inside of the singularity container
JULIAEXE = "/opt/julia-1.11.2/bin/julia"
JULIAFLAGS = ["-J", str(PROJROOT / "bin/julia/nik.so)")]
JULIAFILE = PROJROOT / "bin/julia/sgtsnepi.jl"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n{JULIAFILE}\n")

    name, kwargs = path_to_kwargs(path)
    assert name == "sgtsnepi"

    A = sparse.load_npz(zipf)
    Y, errord = sgtsnepi(A, path=path, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)

        with zf.open("loss.csv", "w") as f:
            f.write(b"iter,error\n")
            for k, v in errord.items():
                f.write(f"{k},{v}\n".encode())


def sgtsnepi(
    A,
    path=None,
    n_epochs=1000,
    early_exaggeration_iter=250,
    dim=2,
    lr=200,
    initialization="random",
    random_state=505**3,
    n_jobs=-1,
    **kwargs,
):

    A = A.tocoo()
    if initialization != "random":
        import warnings

        warnings.warn(
            "If initialization is not ranodm, sgtsnepi might return NaNs.  "
            "Not sure why unfortunately."
        )
    Y0 = _get_init(A, initialization, dim=dim, random_state=random_state)

    # need to create a system call kind of like:

    # julia sgtsnepi.py
    # --row row.npy
    # --col col.npy
    # --data data.npy
    # --shape 2485
    # --init init.npy
    # --outfile $3

    kwargs1 = dict(
        max_iter=n_epochs,
        early_exag=early_exaggeration_iter,
        dim=dim,
        λ=10,
        ɑ=12,
        np=n_jobs if n_jobs != -1 else 0,
        learning_rate=lr,
        shape=A.shape[0],
    )
    kwargs1.update(kwargs)

    args = []
    for k, v in kwargs1.items():
        args += [f"--{k}", f"{v}"]

    with tempfile.TemporaryDirectory(dir=path) as tmp:
        tmpdir = Path(tmp)
        for k in ["row", "col", "data"]:
            np.save(tmpdir / f"{k}.npy", getattr(A, k))
            args += [f"--{k}", str(tmpdir / f"{k}.npy")]

        np.save(tmpdir / "init.npy", Y0)
        args += ["--init", str(tmpdir / "init.npy")]

        args += ["--outfile", f"{tmpdir}/out.npy"]
        julialist = [JULIAEXE, *JULIAFLAGS, str(JULIAFILE)]
        proc = subprocess.run(
            julialist + args,
            check=True,
            capture_output=True,
            encoding="utf8",
        )
        errord = dict()
        for line in proc.stdout.split("\n"):
            m = re.match(r"Iteration (\d+): error is ([^\s]+)", line)
            if m is not None:
                errord[int(m[1])] = float(m[2])
        Y = np.load(tmpdir / "out.npy")

    return Y, errord
