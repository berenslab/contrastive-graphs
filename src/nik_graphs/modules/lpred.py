import inspect
import zipfile

import numpy as np
import polars as pl
from scipy import sparse
from sklearn import metrics

from ..path_utils import path_to_kwargs

__partition__ = "cpu-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "lpred"

    eval_all_embeddings = kwargs.pop("all", False)

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n")

    # assumption: we have the embedding generated in the direct parent
    # and the dataset is defined on directory above.  This should
    # maybe have some code that actually searches the parent dirs, but
    # so far this holds for all approaches, except for the case where
    # we evaluate the feature space of the dataset directly.
    embeddings_dir = path.parent
    data_dir = path.parent.parent
    # if we evaluate the input feature space directly, we change the
    # data_dir and the corresponding embedding key
    if data_dir.name == "runs":
        data_dir = embeddings_dir
        embedding_key = "features"
    else:
        embedding_key = "embedding"

    X = np.load(embeddings_dir / "1.zip")[embedding_key]
    A = sparse.load_npz(data_dir / "1.zip")

    score = link_pred(X, A, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("score.txt", "w") as f:
            f.write(f"{score}\n".encode())

    if eval_all_embeddings:
        df = lpred_other_embeddings(embeddings_dir, A, **kwargs)
        with zipfile.ZipFile(outfile, "a") as zf:
            with zf.open("scores.csv", "w") as f:
                df.write_csv(f)


def link_pred(
    Z,
    A,
    test_size=1000,
    metric="euclidean",
    mode="auc",
    n_jobs=-1,
    random_state=110099,
):

    _, edges, y = graph_train_test(A, test_size=test_size, seed=random_state)

    dists = score_edges(edges[0], edges[1], Z, metric=metric)
    if mode == "auc":
        score = metrics.roc_auc_score(y, dists)
    elif mode == "ap":
        score = metrics.average_precision_score(y, dists)
    else:
        raise ValueError(f"Unkown {mode=!r}, only 'auc', 'ap' allowed")

    return score


def score_edges(rows, cols, embd, metric="cosine"):
    if metric == "cosine":
        norms = np.linalg.norm(embd, axis=1)
        embd = embd / norms[:, None]
        scores = (embd[rows] * embd[cols]).sum(-1)
    elif metric == "euclidean":
        scores = -np.linalg.norm(embd[rows] - embd[cols], axis=1)
    else:
        raise ValueError(f"Unknown {metric=!r}")
    return scores


def lpred_other_embeddings(embeddings_dir, A, **kwargs):
    npz = np.load(embeddings_dir / "1.zip")
    other_embks = [k for k in npz.keys() if k.startswith("embeddings/step-")]
    n = len("embeddings/step-")
    step_keys = [int(k[n:]) for k in other_embks]
    scores = [link_pred(npz[k], A, **kwargs) for k in other_embks]
    df_scores = pl.DataFrame(dict(step=step_keys, score=scores))
    with zipfile.ZipFile(embeddings_dir / "1.zip") as zf:
        if "lightning_logs/steps.csv" in zf.namelist():
            with zf.open("lightning_logs/steps.csv") as f:
                df_epochs = pl.read_csv(f)
        else:
            df_epochs = pl.DataFrame(
                dict(global_step=step_keys, epoch=step_keys)
            )

    return df_scores.join(df_epochs, left_on="step", right_on="global_step")


def graph_train_test(graph, test_size=10_000, seed=0):
    # assumes and unweighted graph
    np.random.seed(seed)
    graph = graph.tocoo()

    assert np.all(graph.data == np.ones(graph.data.shape[0]))
    n_nodes = graph.shape[0]

    # only sample from directed edges to avoid duplicates
    mask = graph.row < graph.col
    directed_rows = graph.row[mask]
    directed_cols = graph.col[mask]

    n_edges = len(directed_rows)
    n_pos_edges = test_size

    pos_test_idx = np.random.choice(
        np.arange(n_edges, dtype=int), n_pos_edges, replace=False
    )
    pos_test_rows, pos_test_cols = (
        directed_rows[pos_test_idx],
        directed_cols[pos_test_idx],
    )
    pos_test_edges = np.stack([pos_test_rows, pos_test_cols])

    # sample negative test edges
    neg_test_rows = []
    neg_test_cols = []
    while len(neg_test_rows) < n_pos_edges:
        neg_rows = np.random.choice(
            np.arange(n_nodes, dtype=int), n_pos_edges, replace=True
        )
        neg_cols = np.random.choice(
            np.arange(n_nodes, dtype=int), n_pos_edges, replace=True
        )

        # ensure uniqueness and avoid self-loops, by only allowing
        # neg_col to be larger than neg_rows
        mask = neg_rows < neg_cols
        neg_rows = neg_rows[mask]
        neg_cols = neg_cols[mask]

        # ensure that neg_rows and neg_cols are not part of the graph
        graph_edges = set(
            zip(graph.row, graph.col)
        )  # graph needs to be symmetric, st both ij and ji are in the
        # graph edges

        prev_neg_edges = set(
            zip(neg_test_rows, neg_test_cols)
        )  # one direction suffices, since we only sample from directed edges

        neg_edges = zip(neg_rows, neg_cols)

        # substract the graph edges and the previous negative edges
        # from the current candidate negative edges
        neg_edges = set(neg_edges).difference(
            graph_edges.union(prev_neg_edges)
        )

        neg_edges = np.stack(list(neg_edges))

        neg_test_rows.extend(neg_edges[:, 0])
        neg_test_cols.extend(neg_edges[:, 1])

    neg_test_edges = np.stack(
        [np.array(neg_test_rows), np.array(neg_test_cols)]
    )
    neg_test_edges = neg_test_edges[
        :, :n_pos_edges
    ]  # remove any extra negative edges

    # remove the pos test edges from the graph
    graph = graph.tocsr()
    graph[pos_test_rows, pos_test_cols] = 0
    graph[pos_test_cols, pos_test_rows] = 0
    graph.eliminate_zeros()
    graph = graph.tocoo()

    assert (
        sparse.csgraph.connected_components(graph, directed=False)[0] == 1
    ), "Graph without positive test edges is not connected anymore."

    # get labels
    y = np.concatenate([np.ones(n_pos_edges), np.zeros(n_pos_edges)]).astype(
        int
    )

    return (
        graph,
        np.concatenate([pos_test_edges, neg_test_edges], axis=-1).astype(int),
        y,
    )
