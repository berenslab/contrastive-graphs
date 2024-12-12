import inspect
from pathlib import Path

from sklearn import datasets

from ..graph_utils import make_adj_mat, save_graph


def run_path(p, outfile):

    # disrespect XDG cache dir
    cache_dir = Path.home() / ".cache/scikit_learn_data"
    mnist = datasets.fetch_openml("mnist_784", data_home=cache_dir)

    adj, annoy_index = make_adj_mat(mnist["data"].values)
    features = mnist["data"].values
    labels = mnist["target"].cat.codes.values
    # will not use `G` here because it's derived from the adjacency matrix
    # G = nx.from_scipy_sparse_array(adj).to_undirected()

    assert not Path(outfile).exists(), f"{outfile} must not exist."
    save_graph(outfile, adj, features, labels)

    with open(p / "files.dep", "a") as f:
        [f.write(inspect.getfile(x) + "\n") for x in [save_graph, datasets]]
