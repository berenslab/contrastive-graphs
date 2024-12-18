from dgl.data import AmazonCoBuyComputerDataset

from .cora import dgl_dataset

__partition__ = "cpu-galvani"


def run_path(p, outfile):
    cls = AmazonCoBuyComputerDataset
    return dgl_dataset(cls, p, outfile)
