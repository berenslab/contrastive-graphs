# Node Embeddings via Neighbor Embeddings

This is the repository accompanying the paper: “Node Embeddings via
Neighbor Embeddings” ([TMLR, 2025](https://arxiv.org/abs/2503.23822)).
It holds all of the code that was used for the experiments, as well as
the code for plotting and constructing the table.

Please cite the following paper:

```bibtex
@misc{boehm2025node,
      title={Node Embeddings via Neighbor Embeddings}, 
      author={Jan Niklas Böhm and Marius Keute and Alica Guzmán and Sebastian Damrich and Andrew Draganov and Dmitry Kobak},
      year={2025},
      journal={Transactions of Machine Learning Research},
}
```

The structure of the repository is as follows:
```
.
├── bin
├── dataframes
├── media
├── runs
└── src/nik_graphs
```

The code is contained in `src/nik_graphs`.  The experiment results are
all stored in a hierarchy within `runs`.  Aggregates of those
experiments are then collected in `dataframes` (which consists mostly
of `.parquet` and `.h5` files).  In `media` all of the output files
are stored.  The folder `bin` holds code to create some binaries that
are used for the experiments or for plotting.  The repository does not
hold any of these files (except for the source code).  You can either
reproduce them by running the experiments or reach out to the authors
to request data files.
