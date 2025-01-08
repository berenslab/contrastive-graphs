import inspect
import zipfile

import numpy as np
from scipy import sparse

from fa2 import ForceAtlas2

from ..path_utils import path_to_kwargs
from .nx import _get_init
from .tsne import TSNECallback

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        ms = [path_to_kwargs, _get_init, TSNECallback]
        [f.write(f"{inspect.getfile(m)}\n") for m in ms]

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "fa2"

    callbacks_every_iters = kwargs.get("callbacks_every_iters", 10)
    callbacks = FA2Callback(outfile, callbacks_every_iters, save_freq=5)
    kwargs["callbacks_every_iters"] = callbacks_every_iters
    kwargs["callbacks"] = callbacks

    sig = inspect.signature(forceatlas2)
    default_init = sig.parameters["initialization"].default
    if kwargs.get("initialization", default_init) == "spectral":
        random_state = kwargs.get("random_state", None)
        if random_state is not None:
            spectral_key = f"spectral/{random_state}"
        else:
            spectral_key = "spectral"
        Y_init = np.load(zipf)[spectral_key][:, :2]
        Y_init /= Y_init[:, 0].std() / 100
        kwargs["initialization"] = Y_init

    Y = forceatlas2(A, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def forceatlas2(
    A,
    n_epochs=100,
    dim=2,
    initialization="spectral",
    random_state=505**3,
    callbacks_every_iters=0,
    callbacks=None,
    verbose=False,
    **kwargs,
):
    Y_init = _get_init(A, initialization, dim=dim, random_state=random_state)

    fa2 = ForceAtlas2(verbose=verbose, **kwargs)
    return fa2.forceatlas2(
        A,
        Y_init,
        iterations=n_epochs,
        callbacks_every_iters=callbacks_every_iters,
        callbacks=callbacks,
    )


class FA2Callback(TSNECallback):
    def __call__(self, iteration, embedding):
        super().__call__(iteration, float("-inf"), embedding)
