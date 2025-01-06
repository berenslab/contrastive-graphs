import inspect
import zipfile

import networkx as nx
import numpy as np
import openTSNE.initialization
from scipy import sparse

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n")

    A = sparse.load_npz(zipf)

    name, kwargs = path_to_kwargs(path)
    assert name == "nx"

    Y = nx_layout(A, **kwargs)

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def nx_layout(
    A,
    layout,
    n_epochs=100,
    dim=2,
    initialization="spectral",
    random_state=505**3,
    **kwargs,
):

    layout_name = f"{layout}_layout"
    layout_function = getattr(nx, layout_name)
    dyn_kwargs = dict()
    match layout:
        case "spring":
            dyn_kwargs["iterations"] = n_epochs
        case "forceatlas2" | "arf":
            dyn_kwargs["max_iter"] = n_epochs

    sig = inspect.signature(layout_function)
    if "pos" in sig.parameters:
        Y_init = _get_init(A, initialization, dim, random_state)
        dyn_kwargs["pos"] = {i: Y_init[i] for i in range(A.shape[0])}
    if "dim" in sig.parameters:
        dyn_kwargs["dim"] = dim
    if "seed" in sig.parameters:
        dyn_kwargs["seed"] = random_state

    G = nx.from_scipy_sparse_array(A)
    pos_dict = layout_function(G, **dyn_kwargs, **kwargs)
    return list(pos_dict.values())


def _get_init(A, initialization, dim, random_state):
    rng = np.random.default_rng(random_state)
    if (
        isinstance(initialization, np.ndarray)
        and initialization.shape[0] == A.shape[0]
    ):
        Y_init = initialization[:, :dim]
    elif initialization == "spectral":
        _init = openTSNE.initialization.spectral(
            sparse.csr_matrix(A).asfptype(),
            n_components=dim,
            random_state=random_state,
            add_jitter=True,
        )
        Y_init = openTSNE.initialization.rescale(
            _init, inplace=True, target_std=1
        )
    elif initialization == "random":
        Y_init = rng.uniform(size=(A.shape[0], dim))
    return Y_init
