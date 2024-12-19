import inspect
import zipfile

import numpy as np
from scipy import sparse

from tfdp.tfdp import tFDP

from ..path_utils import path_to_kwargs
from .nx import _get_init

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n")

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "tfdp"

    Y = tfdp(A, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def tfdp(
    A,
    n_epochs=100,
    initialization="spectral",
    random_state=505**3,
    **kwargs,
):

    Y_init = _get_init(A, initialization, 2, random_state)
    _tfdp = tFDP(
        init=Y_init, max_iter=n_epochs, randseed=random_state, **kwargs
    )

    A = A.tocsr()
    graph = sparse.tril(A)
    edgesrc = A.indptr
    edgetgt = A.indices

    Y, _elapsed_time = _tfdp.optmization(graph, edgesrc, edgetgt)
    return Y
