import inspect
import multiprocessing
import zipfile

import numpy as np
import openTSNE.affinity
from scipy import sparse
from sklearn import preprocessing

from noack import Noack

from ..path_utils import path_to_kwargs
from .nx import _get_init
from .tsne import TSNECallback

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        pyobjs = [path_to_kwargs, TSNECallback, _get_init]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "noack"

    callbacks_every_iters = kwargs.get("callbacks_every_iters", 1)
    callbacks = TSNECallback(outfile, callbacks_every_iters, save_freq=5)
    kwargs["callbacks_every_iters"] = callbacks_every_iters
    kwargs["callbacks"] = callbacks

    Y = noack(A, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def noack(
    A,
    n_epochs=50,
    a=1,
    r=-1,
    n_jobs="auto",
    initialization="spectral",
    random_state=505**3,
    theta=0.9,
    max_grad_norm=100,
    **kwargs,
):
    n_iter = n_epochs
    Y_init = 100 * _get_init(
        A, initialization, dim=2, random_state=random_state
    )
    if n_jobs == "auto":
        n_jobs = min(24, multiprocessing.cpu_count())

    noack = Noack(
        n_jobs=n_jobs,
        n_iter=n_iter,
        initialization=Y_init,
        random_state=random_state,
        theta=0.9,
        max_grad_norm=max_grad_norm,
        **kwargs,
    )

    # normalize affinities row-wise, then symmetrize.  Will be
    # normalized into a joint probability distribution by
    # `PrecomputedAffinities`
    affinities = openTSNE.affinity.PrecomputedAffinities(A.astype(float))
    return noack.fit(affinities=affinities)
