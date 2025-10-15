import inspect
import zipfile

import numpy as np
from scipy import sparse

from ..path_utils import path_to_kwargs


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n")

    name, kwargs = path_to_kwargs(path)

    A = sparse.load_npz(zipf)
    Y = spectral(A, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def spectral(adjacency_mat, dim=2, random_state=1512934):
    from sklearn.manifold import SpectralEmbedding

    affinity_mat = adjacency_mat / adjacency_mat.sum()
    spectral = SpectralEmbedding(
        dim,
        affinity="precomputed",
        eigen_solver="lobpcg",
        n_jobs=-1,
        random_state=random_state,
    )
    return spectral.fit_transform(affinity_mat)
