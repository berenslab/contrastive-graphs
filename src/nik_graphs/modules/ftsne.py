import inspect
import zipfile

import numpy as np

from ..graph_utils import make_adj_mat
from ..path_utils import path_to_kwargs
from .tsne import tsne

__partition__ = "cpu-galvani"


def run_path(path, outfile):

    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        pyobjs = [make_adj_mat, path_to_kwargs, tsne]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]

    name, kwargs = path_to_kwargs(path)
    assert name == "ftsne"
    do_all = kwargs.pop("all", False)

    # A = sparse.load_npz(zipf)
    npz = np.load(zipf)
    if "embedding" in npz:
        features = npz["embedding"]
        data_zip = zipf.parent.parent / "1.zip"
    else:
        features = npz["features"]
        data_zip = zipf

    sig = inspect.signature(tsne)
    default_init = sig.parameters["initialization"].default
    if kwargs.get("initialization", default_init) == "spectral":
        random_state = kwargs.get("random_state", None)
        if random_state is not None:
            spectral_key = f"spectral/{random_state}"
        else:
            spectral_key = "spectral"
        Y_init = np.load(data_zip)[spectral_key][:, :2]
        Y_init /= Y_init[:, 0].std() / 1e-4
        kwargs["initialization"] = Y_init

    Y = feature_tsne(features, **kwargs)

    if do_all:
        tsned = tsne_other_embeddings(zipf.parent, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)

        if do_all:
            for k, v in tsned.items():
                with zf.open(f"{k}.npy", "w") as f:
                    np.save(f, v)


def feature_tsne(
    features, pca_dim=50, initialization="pca", random_state=5015153, **kwargs
):
    rng = np.random.default_rng(random_state)

    if pca_dim < features.shape[1]:
        from sklearn import decomposition

        pca = decomposition.PCA(pca_dim, random_state=rng.integers(2**31 - 1))
        X = pca.fit_transform(features)
    else:
        X = features

    if initialization == "pca":
        from sklearn import decomposition

        pca = decomposition.PCA(2, random_state=rng.integers(2**31 - 1))
        initialization = (
            X[:, :2] if pca_dim < features.shape[1] else pca.fit_transform(X)
        )
        initialization /= initialization.std(axis=0)[0] * 1e4

    if "metric" in kwargs:
        metric = kwargs["metric"]
        _kws = dict(metric=metric if metric != "cosine" else "angular")
    else:
        _kws = dict()
    A, _ = make_adj_mat(X, seed=rng.integers(2**31 - 1), **_kws)

    return tsne(
        A, initialization=initialization, random_state=random_state, **kwargs
    )


def tsne_other_embeddings(embeddings_dir, **kwargs):
    npz = np.load(embeddings_dir / "1.zip")
    other_embks = [k for k in npz.keys() if k.startswith("embeddings/step-")]
    tsnes = [feature_tsne(npz[k], **kwargs) for k in other_embks]
    return {k: t for k, t in zip(other_embks, tsnes)}
