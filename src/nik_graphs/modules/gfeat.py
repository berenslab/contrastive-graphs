import zipfile

import numpy as np
import torch
from scipy import sparse


def run_path(path, outfile):
    torch.set_float32_matmul_precision("medium")
    zipf = path.parent / "1.zip"

    A = sparse.load_npz(zipf)

    name = path.name.split(",")[0]
    assert name == "gfeat"

    Y = A.todense()

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)
