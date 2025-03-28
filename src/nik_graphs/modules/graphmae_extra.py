import argparse
import logging
from functools import partial
from itertools import chain
from typing import Optional

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from torch import optim
from torch.nn import functional as F


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--max_epoch", type=int, default=500, help="number of training epochs"
    )
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="number of hidden attention heads",
    )
    parser.add_argument(
        "--num_out_heads",
        type=int,
        default=1,
        help="number of output attention heads",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of hidden layers"
    )
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--num_remasking", type=int, default=3)
    parser.add_argument(
        "--num_hidden", type=int, default=512, help="number of hidden units"
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        default=False,
        help="use residual connection",
    )
    parser.add_argument(
        "--in_drop", type=float, default=0.2, help="input feature dropout"
    )
    parser.add_argument(
        "--attn_drop", type=float, default=0.1, help="attention dropout"
    )
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="weight decay"
    )
    parser.add_argument(
        "--negative_slope",
        type=float,
        default=0.2,
        help="the negative slope of leaky relu",
    )
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--remask_rate", type=float, default=0.5)
    parser.add_argument("--remask_method", type=str, default="random")
    parser.add_argument(
        "--mask_type", type=str, default="mask", help="`mask` or `drop`"
    )
    parser.add_argument("--mask_method", type=str, default="random")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--drop_edge_rate_f", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2)
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--max_epoch_f", type=int, default=300)
    parser.add_argument("--lr_f", type=float, default=0.01)
    parser.add_argument("--weight_decay_f", type=float, default=0.0)
    parser.add_argument("--linear_prob", action="store_true", default=False)

    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_f", type=int, default=128)
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="saint",
        help="sampling method, `lc` or `saint`",
    )

    parser.add_argument("--label_rate", type=float, default=1.0)
    parser.add_argument("--ego_graph_file_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")

    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument(
        "--full_graph_forward", action="store_true", default=False
    )
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.996)

    args = parser.parse_args()
    return args


def setup_module(
    m_type,
    enc_dec,
    in_dim,
    num_hidden,
    out_dim,
    num_layers,
    dropout,
    activation,
    residual,
    norm,
    nhead,
    nhead_out,
    attn_drop,
    negative_slope=0.2,
    concat_out=True,
    **kwargs,
) -> nn.Module:
    if m_type in ("gat", "tsgat"):
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
            **kwargs,
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden * 2, out_dim),
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int,
        num_dec_layers: int,
        num_remasking: int,
        nhead: int,
        nhead_out: int,
        activation: str,
        feat_drop: float,
        attn_drop: float,
        negative_slope: float,
        residual: bool,
        norm: Optional[str],
        mask_rate: float = 0.3,
        remask_rate: float = 0.5,
        remask_method: str = "random",
        mask_method: str = "random",
        encoder_type: str = "gat",
        decoder_type: str = "gat",
        loss_fn: str = "byol",
        drop_edge_rate: float = 0.0,
        alpha_l: float = 2,
        lam: float = 1.0,
        delayed_ema_epoch: int = 0,
        momentum: float = 0.996,
        replace_rate: float = 0.0,
        zero_init: bool = False,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._remask_rate = remask_rate
        self._mask_method = mask_method
        self._alpha_l = alpha_l
        self._delayed_ema_epoch = delayed_ema_epoch

        self.num_remasking = num_remasking
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._drop_edge_rate = drop_edge_rate
        self._output_hidden_size = num_hidden
        self._momentum = momentum
        self._replace_rate = replace_rate
        self._num_remasking = num_remasking
        self._remask_method = remask_method

        self._token_rate = 1 - self._replace_rate
        self._lam = lam

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat",):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = (
            num_hidden // nhead if decoder_type in ("gat",) else num_hidden
        )

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            nhead_out=nhead_out,
            num_layers=num_dec_layers,
            nhead=nhead,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, num_hidden))

        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        if not zero_init:
            self.reset_parameters_for_token()

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

        self.projector = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )
        self.projector_ema = nn.Sequential(
            nn.Linear(num_hidden, 256),
            nn.PReLU(),
            nn.Linear(256, num_hidden),
        )
        self.predictor = nn.Sequential(
            nn.PReLU(), nn.Linear(num_hidden, num_hidden)
        )

        self.encoder_ema = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.encoder_ema.load_state_dict(self.encoder.state_dict())
        self.projector_ema.load_state_dict(self.projector.state_dict())

        for p in self.encoder_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.projector_ema.parameters():
            p.requires_grad = False
            p.detach_()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(
        self, g, x, targets=None, epoch=0, drop_g1=None, drop_g2=None
    ):  # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(
            g, x, targets, epoch, drop_g1, drop_g2
        )

        return loss

    def mask_attr_prediction(
        self, g, x, targets, epoch, drop_g1=None, drop_g2=None
    ):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            g, x, self._mask_rate
        )
        use_g = drop_g1 if drop_g1 is not None else g

        enc_rep = self.encoder(
            use_g,
            use_x,
        )

        with torch.no_grad():
            drop_g2 = drop_g2 if drop_g2 is not None else g
            latent_target = self.encoder_ema(
                drop_g2,
                x,
            )
            if targets is not None:
                latent_target = self.projector_ema(latent_target[targets])
            else:
                latent_target = self.projector_ema(latent_target[keep_nodes])

        if targets is not None:
            latent_pred = self.projector(enc_rep[targets])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)
        else:
            latent_pred = self.projector(enc_rep[keep_nodes])
            latent_pred = self.predictor(latent_pred)
            loss_latent = sce_loss(latent_pred, latent_target, 1)

        # ---- attribute reconstruction ----
        origin_rep = self.encoder_to_decoder(enc_rep)

        loss_rec_all = 0
        if self._remask_method == "random":
            for i in range(self._num_remasking):
                rep = origin_rep.clone()
                rep, remask_nodes, rekeep_nodes = self.random_remask(
                    use_g, rep, self._remask_rate
                )
                recon = self.decoder(pre_use_g, rep)

                x_init = x[mask_nodes]
                x_rec = recon[mask_nodes]
                loss_rec = self.criterion(x_init, x_rec)
                loss_rec_all += loss_rec
            loss_rec = loss_rec_all
        elif self._remask_method == "fixed":
            rep = self.fixed_remask(g, origin_rep, mask_nodes)
            x_rec = self.decoder(pre_use_g, rep)[mask_nodes]
            x_init = x[mask_nodes]
            loss_rec = self.criterion(x_init, x_rec)
        else:
            raise NotImplementedError

        loss = loss_rec + self._lam * loss_latent

        if epoch >= self._delayed_ema_epoch:
            self.ema_update()
        return loss

    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
                # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(
                    student.parameters(), teacher.parameters()
                ):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        update(self.encoder, self.encoder_ema)
        update(self.projector, self.projector_ema)

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    def get_encoder(self):
        # self.encoder.reset_classifier(out_size)
        return self.encoder

    def reset_encoder(self, out_size):
        self.encoder.reset_classifier(out_size)

    @property
    def enc_params(self):
        return self.encoder.parameters()

    @property
    def dec_params(self):
        return chain(
            *[self.encoder_to_decoder.parameters(), self.decoder.parameters()]
        )

    def output_grad(self):
        grad_dict = {}
        for n, p in self.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad.abs().mean().item()
        return grad_dict

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, g, rep, remask_rate=0.5):

        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[:num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes

    def fixed_remask(self, g, rep, masked_nodes):
        rep[masked_nodes] = 0
        return rep


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


class GAT(nn.Module):
    def __init__(
        self,
        in_dim,
        num_hidden,
        out_dim,
        num_layers,
        nhead,
        nhead_out,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        norm,
        concat_out=False,
        encoding=False,
    ):
        super(GAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_heads_out = nhead_out
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        hidden_in = in_dim
        hidden_out = out_dim

        if num_layers == 1:
            self.gat_layers.append(
                GATConv(
                    hidden_in,
                    hidden_out,
                    nhead_out,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    last_residual,
                    norm=last_norm,
                    concat_out=concat_out,
                )
            )
        else:
            # input projection (no residual)
            self.gat_layers.append(
                GATConv(
                    hidden_in,
                    num_hidden,
                    nhead,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    create_activation(activation),
                    norm=norm,
                    concat_out=concat_out,
                )
            )
            # hidden layers

            for _ in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(
                    GATConv(
                        num_hidden * nhead,
                        num_hidden,
                        nhead,
                        feat_drop,
                        attn_drop,
                        negative_slope,
                        residual,
                        create_activation(activation),
                        norm=norm,
                        concat_out=concat_out,
                    )
                )

            # output projection
            self.gat_layers.append(
                GATConv(
                    num_hidden * nhead,
                    hidden_out,
                    nhead_out,
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    last_residual,
                    activation=last_activation,
                    norm=last_norm,
                    concat_out=concat_out,
                )
            )
        self.head = nn.Identity()

    def forward(self, g, inputs):
        h = inputs

        for i in range(self.num_layers):
            h = self.gat_layers[i](g, h)

        if self.head is not None:
            return self.head(h)
        else:
            return h

    def inference(self, g, x, batch_size, device, emb=False):
        """Inference with the GAT model on full neighbors
        (i.e. without neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.  The inference code is
        written in a fashion that it could handle any number of nodes
        and layers.

        """
        num_heads = self.num_heads
        num_heads_out = self.num_heads_out
        for i, layer in enumerate(self.gat_layers):
            if i < self.num_layers - 1:
                y = torch.zeros(
                    g.num_nodes(),
                    (
                        self.num_hidden * num_heads
                        if i != len(self.gat_layers) - 1
                        else self.num_classes
                    ),
                )
            else:
                if emb is False:
                    y = torch.zeros(
                        g.num_nodes(),
                        (
                            self.num_hidden
                            if i != len(self.gat_layers) - 1
                            else self.num_classes
                        ),
                    )
                else:
                    y = torch.zeros(
                        g.num_nodes(), self.out_dim * num_heads_out
                    )
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=8,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)
                h = x[input_nodes].to(device)
                if i < self.num_layers - 1:
                    h = layer(block, h)
                else:
                    h = layer(block, h)

                if i == len(self.gat_layers) - 1 and (emb is False):
                    h = self.head(h)
                y[output_nodes] = h.cpu()
            x = y
        return y

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.is_pretraining = False
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        norm=None,
        concat_out=True,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._concat_out = concat_out

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(num_heads * out_feats,))
            )
        else:
            self.register_buffer("bias", None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False
                )
            else:
                self.res_fc = None
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self.activation = activation

        self.norm = norm
        if norm is not None:
            self.norm = create_norm(norm)(num_heads * out_feats)
        self.set_allow_zero_in_degree(False)

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot
        uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                # h_dst = self.feat_drop(feat[1])
                h_dst = feat[1]

                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (
                        graph.number_of_dst_nodes(),
                    ) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j,
            # respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))
            # e[e == 0] = -1e3
            # e = graph.edata.pop('e')
            # compute softmax
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)),
                    self._num_heads,
                    self._out_feats,
                )

            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval

            if self._concat_out:
                rst = rst.flatten(1)
            else:
                rst = torch.mean(rst, dim=1)

            if self.norm is not None:
                rst = self.norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x

    return func


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        # print("Identity norm")
        return None


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    remask_rate = args.remask_rate
    mask_method = args.mask_method
    drop_edge_rate = args.drop_edge_rate

    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l

    num_features = args.num_features
    num_dec_layers = args.num_dec_layers
    num_remasking = args.num_remasking
    lam = args.lam
    delayed_ema_epoch = args.delayed_ema_epoch
    replace_rate = args.replace_rate
    remask_method = args.remask_method
    momentum = args.momentum
    zero_init = args.dataset in ("cora", "pubmed", "citeseer")

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        num_dec_layers=num_dec_layers,
        num_remasking=num_remasking,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        remask_rate=remask_rate,
        mask_method=mask_method,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        alpha_l=alpha_l,
        lam=lam,
        delayed_ema_epoch=delayed_ema_epoch,
        replace_rate=replace_rate,
        remask_method=remask_method,
        momentum=momentum,
        zero_init=zero_init,
    )
    return model


def create_optimizer(
    opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None
):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


def pretrain(
    model,
    feats,
    graph,
    ego_graph_nodes,
    max_epoch,
    device,
    use_scheduler,
    lr,
    weight_decay,
    batch_size=512,
    sampling_method="lc",
    optimizer="adam",
    drop_edge_rate=0,
):
    logging.info("start training..")

    model = model.to(device)
    optimizer = create_optimizer(optimizer, model, lr, weight_decay)

    dataloader = setup_training_dataloder(
        sampling_method,
        ego_graph_nodes,
        graph,
        feats,
        batch_size=batch_size,
        drop_edge_rate=drop_edge_rate,
    )

    if use_scheduler and max_epoch > 0:
        logging.info("Use scheduler")
        scheduler = (
            lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=scheduler
        )
    else:
        scheduler = None

    for epoch in range(max_epoch):
        epoch_iter = iter(dataloader)
        losses = []
        # assert (graph.in_degrees() > 0).all(), "after loading"

        for batch_g in epoch_iter:
            model.train()
            if drop_edge_rate > 0:
                batch_g, targets, _, node_idx, drop_g1, drop_g2 = batch_g
                batch_g = batch_g.to(device)
                drop_g1 = drop_g1.to(device)
                drop_g2 = drop_g2.to(device)
                x = batch_g.ndata.pop("feat")
                loss = model(batch_g, x, targets, epoch, drop_g1, drop_g2)
            else:
                batch_g, targets, _, node_idx = batch_g
                batch_g = batch_g.to(device)
                x = batch_g.ndata.pop("feat")
                loss = model(batch_g, x, targets, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        # torch.save(model.state_dict(), os.path.join(model_dir, model_name))

    return model


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


class OnlineLCLoader(torch.utils.data.DataLoader):
    def __init__(
        self, root_nodes, graph, feats, labels=None, drop_edge_rate=0, **kwargs
    ):
        self.graph = graph
        self.labels = labels
        self._drop_edge_rate = drop_edge_rate
        self.ego_graph_nodes = root_nodes
        self.feats = feats

        dataset = np.arange(len(root_nodes))
        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset, **kwargs)

    def drop_edge(self, g):
        if self._drop_edge_rate <= 0:
            return g, g

        g = g.remove_self_loop()
        mask_index1 = mask_edge(g, self._drop_edge_rate)
        mask_index2 = mask_edge(g, self._drop_edge_rate)
        g1 = dgl.remove_edges(g, mask_index1).add_self_loop()
        g2 = dgl.remove_edges(g, mask_index2).add_self_loop()
        return g1, g2

    def __collate_fn__(self, batch_idx):
        ego_nodes = [self.ego_graph_nodes[i] for i in batch_idx]
        subgs = [
            self.graph.subgraph(ego_nodes[i]) for i in range(len(ego_nodes))
        ]
        sg = dgl.batch(subgs)

        nodes = torch.from_numpy(np.concatenate(ego_nodes)).long()
        num_nodes = [x.shape[0] for x in ego_nodes]
        cum_num_nodes = np.cumsum([0] + num_nodes)[:-1]

        if self._drop_edge_rate > 0:
            drop_g1, drop_g2 = self.drop_edge(sg)

        sg = sg.remove_self_loop().add_self_loop()
        sg.ndata["feat"] = self.feats[nodes]
        targets = torch.from_numpy(cum_num_nodes)

        if self.labels is not None:
            label = self.labels[batch_idx]
        else:
            label = None

        if self._drop_edge_rate > 0:
            return sg, targets, label, nodes, drop_g1, drop_g2
        else:
            return sg, targets, label, nodes


def setup_training_dataloder(
    loader_type,
    training_nodes,
    graph,
    feats,
    batch_size,
    drop_edge_rate=0,
    pretrain_clustergcn=False,
    cluster_iter_data=None,
):
    num_workers = 8

    if loader_type == "lc":
        assert training_nodes is not None
    else:
        raise NotImplementedError(f"{loader_type} is not implemented yet")

    # print(" -------- drop edge rate: {} --------".format(drop_edge_rate))
    dataloader = OnlineLCLoader(
        training_nodes,
        graph,
        feats=feats,
        drop_edge_rate=drop_edge_rate,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        persistent_workers=True,
        num_workers=num_workers,
    )
    return dataloader
