import zipfile

import numpy as np
import openTSNE
from scipy import sparse

from ..path_utils import path_to_kwargs


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{zipf}\n")

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "tsne"
    Y = tsne(A, **kwargs)
    # with zipfile.ZipFile(parent / "1.zip") as zf:

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
        **kwargs,
    )

    A /= A.sum(1)
    A /= A.sum()
    affinities = openTSNE.affinity.PrecomputedAffinities(A, normalize=False)
    return tsne.fit(affinities=affinities)
