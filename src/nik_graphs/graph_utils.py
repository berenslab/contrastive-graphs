import zipfile

import annoy
import numpy as np
from scipy import sparse


def make_adj_mat(
    X,
    n_neighbors=15,
    metric="euclidean",
    n_trees=50,
    seed=None,
    use_dists=False,
    symmetrize=True,
    drop_first=True,
    n_jobs=-1,
):
    t = annoy.AnnoyIndex(X.shape[1], metric)
    if seed is not None:
        t.set_seed(seed)

    [t.add_item(i, x) for i, x in enumerate(X)]
    t.build(n_trees, n_jobs=n_jobs)

    # construct the adjacency matrix for the graph
    adj = sparse.lil_matrix((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        neighs_, dists_ = t.get_nns_by_item(
            i, n_neighbors + 1, include_distances=True
        )
        if drop_first:
            neighs = neighs_[1:]
            dists = dists_[1:]
        else:
            neighs = neighs_[:n_neighbors]
            dists = dists_[:n_neighbors]

        adj[i, neighs] = dists if use_dists else 1
        if symmetrize:
            adj[neighs, i] = dists if use_dists else 1  # symmetrize on the fly

    return adj, t


def save_graph(outfile, adjacency_mat, features, labels):
    # Will save the adjacency matrix `adj` in the outfile directly.
    # This way it can be loaded from the zip archive without first
    # opening the zip file and then opening the file pointer within
    # the archive.
    with open(outfile, "xb") as f:
        sparse.save_npz(f, adjacency_mat)

    # need write mode "a" because we have stored `adj` already
    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("features.npy", "w") as f:
            np.save(f, features)
        with zf.open("labels.npy", "w") as f:
            np.save(f, labels)
