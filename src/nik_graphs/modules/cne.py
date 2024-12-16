import contextlib
import inspect
import logging
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import lightning
import numpy as np
import torch
import tsimcne
from scipy import sparse
from sklearn import (
    linear_model,
    model_selection,
    neighbors,
    pipeline,
    preprocessing,
)

from ..path_utils import path_to_kwargs

__partition__ = "2080-galvani"


def run_path(path, outfile):
    torch.set_float32_matmul_precision("medium")
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        pyobjs = [
            contextlib,
            logging,
            inspect,
            subprocess,
            warnings,
            zipfile,
            Path,
            lightning,
            np,
            torch,
            tsimcne,
            sparse,
            linear_model,
            model_selection,
            neighbors,
            pipeline,
            preprocessing,
        ]
        [f.write(inspect.getfile(x) + "\n") for x in pyobjs]

    A = sparse.load_npz(zipf)
    labels = np.load(zipf)["labels"]

    [
        logging.getLogger(name).setLevel(logging.ERROR)
        for name in logging.root.manager.loggerDict
        if "lightning" in name
    ]
    logger = lightning.pytorch.loggers.CSVLogger(
        save_dir=path, name=None, version=0
    )
    trainer_kwargs = dict(
        log_every_n_steps=50,
        # val_check_interval=20,
        check_val_every_n_epoch=25,
        precision="bf16-mixed",
        enable_model_summary=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )

    name, kwargs = path_to_kwargs(path)
    assert name == "cne"
    Y = tsimcne_nonparam(
        A, labels, trainer_kwargs=trainer_kwargs, logger=logger, **kwargs
    )
    # with zipfile.ZipFile(parent / "1.zip") as zf:

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)

        for p in Path(logger.log_dir).iterdir():
            zf.write(p, f"lightning_logs/{p.name}")
            p.unlink()
        Path(logger.log_dir).rmdir()


def tsimcne_nonparam(
    A,
    labels=None,
    metric="euclidean",
    batch_size="auto",
    trainer_kwargs=None,
    logger=None,
    n_epochs=100,
    temp=0.01,
    eval_ann=False,
    opt="adam",
    lr=1,
    dim=128,
    initial_dim=128,
    weight_decay=0,
    warmup_epochs=0,
    drop_last=True,
    random_state=4101661632,
    **kwargs,
):
    assert initial_dim >= dim
    if trainer_kwargs is None:
        trainer_kwargs = dict()

    torch.manual_seed(random_state)

    if batch_size == "auto":
        batch_size = 2**10 if A.shape[0] < 10_000 else 2**13
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
    with contextlib.redirect_stdout(sys.stderr):
        mod = tsimcne.PLtSimCNE(
            backbone=torch.nn.Embedding(len(y), initial_dim),
            backbone_dim=initial_dim,
            projection_head=torch.nn.Identity(),
            n_epochs=n_epochs,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            metric=metric,
            temperature=temp,
            optimizer_name=opt,
            lr=lr,
            weight_decay=weight_decay,
            out_dim=initial_dim,
            anneal_to_dim=dim,
            # save_intermediate_feat=True,
            batches_per_epoch=len(dm.train_dataloader()),
            eval_ann=eval_ann,
            # eval_function=EvalCB(A),
            **kwargs,
        )
        trainer = lightning.Trainer(
            max_epochs=n_epochs, logger=logger, **trainer_kwargs
        )
        with warnings.catch_warnings():
            msg = (
                "Trying to infer the `batch_size` "
                r"from an ambiguous collection\..*"
            )
            warnings.filterwarnings(action="ignore", message=msg)
            trainer.fit(mod, datamodule=dm)
        if not isinstance(mod.backbone, torch.nn.Embedding):
            trainer.save_checkpoint(Path(trainer.log_dir) / "cne.ckpt")
        out_batches = trainer.predict(mod, datamodule=dm)
    return torch.vstack([x[0] for x in out_batches]).cpu().numpy()


class GraphDM(lightning.LightningDataModule):
    def __init__(self, neigh_mat, labels=None, **kwargs):
        super().__init__()
        self.neigh_mat = neigh_mat
        self.labels = labels
        self.kwargs = kwargs

        proc = subprocess.run("nproc", capture_output=True, text=True)
        self.n_procs = int(proc.stdout)

    def train_dataloader(self):
        return FastTensorDataLoader(
            self.neigh_mat, shuffle=True, **self.kwargs
        )

    def predict_dataloader(self):
        kwargs = self.kwargs.copy()
        kwargs.pop("drop_last", None)
        kwargs.pop("data_on_gpu", None)
        y = (
            range(self.neigh_mat.shape[0])
            if self.labels is None
            else self.labels
        )

        ds = [(i, lbl) for i, lbl in zip(range(self.neigh_mat.shape[0]), y)]
        return torch.utils.data.DataLoader(
            ds, shuffle=False, num_workers=self.n_procs, **kwargs
        )

    def val_dataloader(self):
        bs = self.kwargs["batch_size"]
        val_dl = FastTensorDataLoader(
            self.neigh_mat[:bs], shuffle=False, **self.kwargs
        )

        return [val_dl, self.predict_dataloader()]


# FastTensorDataLoader based on
# https://github.com/berenslab/contrastive-ne/
# blob/1998d338a56f43191d1a3340896dced638054712/src/cne/_cne.py#L76C12-L77C1
# and
# https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching
# /27014/6


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """

    def __init__(
        self,
        neighbor_mat,
        batch_size=1024,
        shuffle=False,
        data_on_gpu=False,
        drop_last=False,
        seed=None,
    ):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param data_on_gpu: If True, the dataset is loaded on GPU as a whole.
        :param drop_last: Drop the last batch if it is smaller than the others.
        :param seed: Random seed

        :returns: A FastTensorDataLoader.
        """

        neighbor_mat = neighbor_mat.tocoo()
        tensors = [
            torch.tensor(neighbor_mat.row),
            torch.tensor(neighbor_mat.col),
        ]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        # manage device
        if data_on_gpu:
            self.device = "cuda"
            tensors = [tensor.to(self.device) for tensor in tensors]
        else:
            self.device = "cpu"
        self.tensors = tensors

        self.dataset_len = torch.tensor(
            self.tensors[0].shape[0], device=self.device
        )
        self.batch_size = torch.tensor(batch_size, dtype=int).to(self.device)

        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # Calculate number of  batches
        n_batches = torch.div(
            self.dataset_len, self.batch_size, rounding_mode="floor"
        )
        remainder = torch.remainder(self.dataset_len, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        else:
            self.indices = None
        self.i = torch.tensor(0, device=self.device)
        return self

    def __next__(self):
        if (
            self.i > self.dataset_len - self.batch_size and self.drop_last
        ) or self.i >= self.dataset_len:
            raise StopIteration

        start = self.i
        end = torch.minimum(self.i + self.batch_size, self.dataset_len)
        if self.indices is not None:
            indices = self.indices[start:end]
            batch = tuple(
                torch.index_select(t, 0, indices) for t in self.tensors
            )
        else:
            batch = tuple(t[start:end] for t in self.tensors)
        self.i += self.batch_size
        dummy_labels = torch.ones(len(batch[0]))
        return torch.concat(batch), dummy_labels

    def __len__(self):
        return self.n_batches


class EvalCB:
    def __init__(self, A):
        self.A = A
        self.eval_counter = 0

    def __call__(self, *, Z, H, labels, step, epoch):
        self.eval_counter += 1
        if epoch < 2 or self.eval_counter % 5 == 0:
            knn, lin, recall = evaluate(
                Z, labels, self.A, metric="cosine", test_size=1000
            )
            return dict(knn=knn, lin=lin, recall=recall)
        else:
            return dict()


def graph_knn_recall(Z, A, test_size=1000, random_state=None, metric="cosine"):
    test_ind = np.random.default_rng(random_state).choice(
        Z.shape[0],
        size=test_size,
        replace=False,
    )

    A_array = A[test_ind].toarray().astype(int)
    max_edges = A_array.sum(axis=1).max().astype(int)

    neigh = neighbors.NearestNeighbors(
        metric=metric, n_neighbors=max_edges + 1
    )
    neigh.fit(Z)

    # Excluding point itself as a neighbor
    Z_neighb = neigh.kneighbors(Z[test_ind], return_distance=False)[:, 1:]

    fraction = 0
    for i in range(test_size):
        set1 = set(Z_neighb[i, : A_array[i].sum()])  # neighbors in Z (embed)
        set2 = set(np.where(A_array[i] > 0)[0])  # neighbors in A (true)
        fraction += len(set1 & set2) / len(set2)  # overlap
    fraction /= test_size

    return fraction


def accuracy(Z, y, random_state=None, metric="cosine"):
    lin = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(penalty=None, max_iter=100),
    )
    knn = neighbors.KNeighborsClassifier(n_neighbors=10, metric=metric)

    # to make it faster, limit test size to 1000
    if Z.shape[0] > 10_000:
        test_size = 1000
    else:
        test_size = 0.1

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        Z, y, test_size=test_size, random_state=random_state
    )

    # Suppress annoying convergence warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        linacc = lin.fit(X_train, y_train).score(X_test, y_test)

    knnacc = knn.fit(X_train, y_train).score(X_test, y_test)

    return linacc, knnacc


def evaluate(Z, y, A, random_state=None, metric="cosine", test_size=1000):
    recall = graph_knn_recall(
        Z, A, test_size=test_size, random_state=random_state, metric=metric
    )
    linacc, knnacc = accuracy(Z, y, random_state=random_state, metric=metric)

    return linacc, knnacc, recall
