from dgl.data import AmazonCoBuyPhotoDataset

from .cora import dgl_dataset

__partition__ = "cpu-galvani"


def run_path(p, outfile):
    cls = AmazonCoBuyPhotoDataset
    return dgl_dataset(cls, p, outfile)
