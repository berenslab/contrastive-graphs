import inspect
import zipfile

import numpy as np
import openTSNE
import openTSNE.callbacks
from scipy import sparse
from sklearn import preprocessing

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        pyobjs = [path_to_kwargs]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "tsne"

    callbacks_every_iters = kwargs.get("callbacks_every_iters", 10)
    callbacks = TSNECallback(outfile, callbacks_every_iters, save_freq=5)
    kwargs["callbacks_every_iters"] = callbacks_every_iters
    kwargs["callbacks"] = callbacks

    sig = inspect.signature(tsne)
    default_init = sig.parameters["initialization"].default
    if kwargs.get("initialization", default_init) == "spectral":
        random_state = kwargs.get("random_state", None)
        if random_state is not None:
            spectral_key = f"spectral/{random_state}"
        else:
            spectral_key = "spectral"
        Y_init = np.load(zipf)[spectral_key][:, :2]
        Y_init /= Y_init[:, 0].std() / 1e-4
        kwargs["initialization"] = Y_init

    Y = tsne(A, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def tsne(
    A,
    n_epochs=750,
    early_exaggeration_iter=None,
    n_jobs=-1,
    initialization="spectral",
    random_state=505**3,
    row_norm=True,
    negative_gradient_method="auto",
    **kwargs,
):
    if early_exaggeration_iter is None:
        n_iter = n_epochs * 2 // 3
        early_exaggeration_iter = n_epochs // 3
    else:
        n_iter = n_epochs
    tsne = openTSNE.TSNE(
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        initialization=initialization,
        random_state=random_state,
        negative_gradient_method=negative_gradient_method,
        **kwargs,
    )

    # normalize affinities row-wise, then symmetrize.  Will be
    # normalized into a joint probability distribution by
    # `PrecomputedAffinities`
    P = preprocessing.normalize(A, norm="l1", axis=1) if row_norm else A
    affinities = openTSNE.affinity.PrecomputedAffinities((P + P.T) / 2)
    return tsne.fit(affinities=affinities)


class TSNECallback(openTSNE.callbacks.Callback):
    def __init__(self, zipfname, callbacks_every_iters, save_freq=5):
        super().__init__()
        self.zipfname = zipfname
        self.callbacks_every_iters = callbacks_every_iters
        self.save_freq = save_freq
        self.counter = 0
        self.n_called = 0
        self.errors = []

    def __call__(self, iteration, error, embedding):

        if (self.n_called + 1) % self.save_freq == 0:
            with zipfile.ZipFile(self.zipfname, "a") as zf:
                fname = f"embeddings/step-{self.counter:05d}.npy"
                with zf.open(fname, "w") as f:
                    np.save(f, embedding.astype("float32"))

        # tsne calls the callback functions if:
        # (iter + 1) % callbacks_every_iters == 0
        self.counter += self.callbacks_every_iters
        self.n_called += 1
        self.errors.append(error)
