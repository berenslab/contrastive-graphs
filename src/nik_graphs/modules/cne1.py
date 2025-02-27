import contextlib
import sys
import zipfile

import numpy as np
import torch
from openTSNE.initialization import rescale, spectral
from scipy import sparse

from cne._cne import CNE

from ..path_utils import path_to_kwargs
from .cne import GraphDM

__partition__ = "2080-galvani"


def run_path(path, outfile):
    torch.set_float32_matmul_precision("medium")
    zipf = path.parent / "1.zip"

    A = sparse.load_npz(zipf)
    npz = np.load(zipf)
    labels = npz["labels"]

    name, kwargs = path_to_kwargs(path)
    assert name == "cne1"

    Y = cne(A, labels, **kwargs)
    # with zipfile.ZipFile(parent / "1.zip") as zf:

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)


def cne(
    A,
    labels=None,
    loss="infonce",
    metric="cosine",
    batch_size="auto",
    negative_samples="full-batch",
    n_epochs=100,
    temp=0.5,
    learn_temp=False,
    opt="adam",
    lr=1,
    dim=128,
    weight_decay=0,
    warmup_epochs=0,
    anneal_lr=True,
    drop_last=True,
    initialization="spectral",
    random_state=4101661632,
    **kwargs,
):

    torch.manual_seed(random_state)

    if batch_size == "auto":
        batch_size = min(A.nnz // 10, 2**13)

    if labels is None:
        y = torch.zeros(A.shape[0], dtype=int)
    else:
        y = labels

    dm = GraphDM(
        A,
        labels=y,
        batch_size=batch_size,
        drop_last=drop_last,
        data_on_gpu=True,
    )

    if (
        isinstance(initialization, np.ndarray)
        and initialization.shape[0] == A.shape[0]
    ):
        # leave it as is
        backbone = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(initialization), freeze=False
        )
    elif initialization == "spectral":
        X = spectral(
            sparse.csr_matrix(A).astype("float32"),
            n_components=dim,
            random_state=random_state,
        )
        init = rescale(
            X,
            inplace=True,
            target_std=1,
        ).astype("float32")
        backbone = torch.nn.Embedding.from_pretrained(
            torch.from_numpy(init), freeze=False
        )
    elif initialization == "random":
        backbone = torch.nn.Embedding(A.shape[0], dim)
    else:
        raise ValueError(f"Wrong {initialization=!r} passed")
    with contextlib.redirect_stdout(sys.stderr):

        cne = CNE(
            model=backbone.cuda(),
            seed=random_state,
            loss_mode=loss,
            metric=metric,
            learning_rate=lr,
            learn_temp=learn_temp,
            negative_samples=negative_samples,
            weight_decay=weight_decay,
            temperature=temp,
            optimizer=opt,
            anneal_lr=anneal_lr,
            **kwargs,
        )

        cne.fit(X=A[:, :2].todense(), graph=A)
    return cne.model.embedding_.detach().cpu()
