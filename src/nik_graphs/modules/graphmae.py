import inspect
import zipfile

import dgl
import numpy as np
from scipy import sparse

from ..path_utils import path_to_kwargs
from .graphmae_extra import build_args, build_model, pretrain

__partition__ = "2080-galvani"


def run_path(path, outfile):
    name, kwargs = path_to_kwargs(path)
    assert name == "graphmae"

    zipf = path.parent / "1.zip"
    adj = sparse.load_npz(zipf)
    npz = np.load(zipf)
    feat = npz["features"]

    Y = graphmae(adj, feat, **kwargs)

    with zipfile.ZipFile(outfile, "x") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)

    with open(path / "files.dep", "a") as f:
        f.write(inspect.getfile(build_model) + "\n")


def graphmae(
    adj, feat, n_epochs=100, opt="adam", random_state=50123, device="cuda:0"
):

    A = adj.tocoo()

    graph = dgl.DglGraph((A.row, A.col))
    graph = dgl.add_self_loop(graph)

    args = build_args()
    args.num_features = feat.shape[1]

    model = build_model(args)

    seeds = [random_state]
    dataset_name = args.dataset
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    encoder = args.encoder
    decoder = args.decoder
    num_hidden = args.num_hidden
    drop_edge_rate = args.drop_edge_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    no_pretrain = args.no_pretrain
    logs = args.logging
    use_scheduler = args.scheduler
    batch_size = args.batch_size
    batch_size_f = args.batch_size_f
    sampling_method = args.sampling_method
    ego_graph_file_path = args.ego_graph_file_path
    data_dir = args.data_dir
    model = pretrain(
        model,
        feat,
        graph,
        pretrain_ego_graph_nodes,
        max_epoch=n_epochs,
        device=device,
        use_scheduler=use_scheduler,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        drop_edge_rate=drop_edge_rate,
        sampling_method=sampling_method,
        optimizer=opt,
    )
