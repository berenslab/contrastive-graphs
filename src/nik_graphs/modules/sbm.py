import inspect
import os
from pathlib import Path

import numpy as np

from ..graph_utils import save_dataset_split, save_graph
from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(p, outfile):

    name, kwargs = path_to_kwargs(p)
    assert name == "sbm"

    cache_dir = os.environ.get("XDG_CACHE_DIR", Path.home() / ".cache")
    data_dir = Path(cache_dir) / "stochastic_block_model"
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(p / "files.dep", "a") as f:
        pyobjs = [path_to_kwargs, save_graph]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]

    return sbm(outfile=outfile, **kwargs)


def sbm(
    n_pts=101,
    n_blocks=10,
    p_intra=0.1,
    p_inter=0.01,
    random_state=1576482771,
    outfile=None,
):
    import networkx as nx
    import numpy as np

    rng = np.random.default_rng(random_state)

    if p_intra <= p_inter:
        import warnings

        warnings.warn(
            f"{p_intra=} is smaller than {p_inter=}, results may be weird."
        )

    block_sizes = [n_pts] * n_blocks
    block_probs = [
        [p_inter] * i + [p_intra] + [p_inter] * (n_blocks - (i + 1))
        for i in range(n_blocks)
    ]
    G = nx.stochastic_block_model(block_sizes, block_probs).to_undirected()
    adj = nx.adjacency_matrix(G).astype("uint8")
    features = np.ones(adj.shape[0], dtype="uint8")
    labels = np.repeat(range(n_blocks), n_pts)

    if outfile is not None:
        assert not Path(outfile).exists(), f"{outfile} must not exist."
        save_graph(
            outfile,
            adj,
            features,
            labels,
            save_spectral=True,
            random_state=rng.integers(2**31 - 1),
        )

    n = adj.shape[0]
    train_size = n * 8 // 10
    test_size = n // 10
    # val_size = n // 10
    inds = rng.permutation(n).astype("uint32")
    train_inds = inds[:train_size]
    test_inds = inds[train_size : train_size + test_size]
    val_inds = inds[train_size + test_size :]
    save_dataset_split(outfile, train_inds, test_inds, val_inds)

    return adj, labels
