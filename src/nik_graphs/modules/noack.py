import inspect
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
    n_jobs=-1,
    initialization="spectral",
    random_state=505**3,
    theta=0.9,
    **kwargs,
):
    n_iter = n_epochs
    rng = np.random.default_rng(random_state)
    Y_init = _get_init(A, initialization, dim=2, rng=rng)

    noack = Noack(
        n_jobs=n_jobs,
        n_iter=n_iter,
        initialization=Y_init,
        random_state=random_state,
        theta=0.9,
        **kwargs,
    )

    # normalize affinities row-wise, then symmetrize.  Will be
    # normalized into a joint probability distribution by
    # `PrecomputedAffinities`
    affinities = openTSNE.affinity.PrecomputedAffinities(A.astype(float))
    return noack.fit(affinities=affinities)
