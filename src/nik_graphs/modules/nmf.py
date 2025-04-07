import inspect
import zipfile

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n")

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "nmf"

    sig = inspect.signature(nmf)
    default_init = sig.parameters["initialization"].default
    if kwargs.get("initialization", default_init) == "spectral":
        random_state = kwargs.get("random_state", None)
        if random_state is not None:
            spectral_key = f"spectral/{random_state}"
        else:
            spectral_key = "spectral"
        kwargs["initialization"] = np.load(zipf)[spectral_key][:, :2]

    Y = nmf(A, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def nmf(
    A,
    n_epochs=100,
    initialization="random",
    random_state=505**3,
    dim=2,
    tol=1e-3,
    solver="md",
    **kwargs,
):

    nmf = NMF(
        n_components=dim,
        init=initialization,
        tol=tol,
        max_iter=n_epochs,
        random_state=random_state,
        **kwargs,
    )

    A = A.tocsr()

    return nmf.fit_transform(A)
