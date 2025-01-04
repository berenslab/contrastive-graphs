from dgl.data import CiteseerGraphDataset

from .cora import dgl_dataset

__partition__ = "cpu-galvani"


def run_path(p, outfile):
    cls = CiteseerGraphDataset
    return dgl_dataset(cls, p, outfile)
