from dgl.data import PubmedGraphDataset

from .cora import dgl_dataset

__partition__ = "cpu-galvani"


def run_path(p, outfile):
    cls = PubmedGraphDataset
    return dgl_dataset(cls, p, outfile)
