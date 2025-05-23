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


def save_graph(
    outfile,
    adjacency_mat,
    features,
    labels,
    save_spectral=True,
    random_state=None,
):
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

        if save_spectral:
            from sklearn.manifold import SpectralEmbedding

            affinity_mat = adjacency_mat / adjacency_mat.sum()
            spectral = SpectralEmbedding(
                128,
                affinity="precomputed",
                eigen_solver="lobpcg",
                n_jobs=-1,
                random_state=random_state,
            )
            X_spectral = spectral.fit_transform(affinity_mat)

            with zf.open("spectral.npy", "w") as f:
                np.save(f, X_spectral)
            for r in [1111, 2222, 3333, 4444]:
                spectral = spectral.set_params(random_state=r)
                X_spectral = spectral.fit_transform(affinity_mat)
                with zf.open(f"spectral/{r}.npy", "w") as f:
                    np.save(f, X_spectral)

        with zf.open("drgraph.txt", "w") as f:

            def write(s):
                return f.write(s.encode())

            A = adjacency_mat.tocoo()
            write(f"{A.shape[0]} {A.nnz}\n")
            [write(f"{r} {c} {v}\n") for r, c, v in zip(A.row, A.col, A.data)]


def save_dataset_split(outfile, train_ind, test_ind, val_ind, prefix="split"):

    inds = dict(train=train_ind, test=test_ind, val=val_ind)
    for name, ar in inds.items():
        with zipfile.ZipFile(outfile, "a") as zf:
            with zf.open(f"{prefix}/{name}.npy", "w") as f:
                np.save(f, ar)


def save_public_dataset_split(*args):
    return save_dataset_split(*args, prefix="split/public")
