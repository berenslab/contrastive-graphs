import inspect
import zipfile

import numpy as np

from ..path_utils import path_to_kwargs


def run_path(path, outfile):
    zipf = path.parent / "1.zip"

    with open(path / "files.dep", "a") as f:
        f.write(f"{inspect.getfile(path_to_kwargs)}\n")

    name, kwargs = path_to_kwargs(path)

    random_state = kwargs.get("random_state", None)
    dim = kwargs.get("dim", 2)
    if random_state is not None:
        spectral_key = f"spectral/{random_state}"
    else:
        spectral_key = "spectral"
    Y = np.load(zipf)[spectral_key][:, :dim]

    with zipfile.ZipFile(outfile, "a") as zf:
        with zf.open("embedding.npy", "w") as f:
            np.save(f, Y)
