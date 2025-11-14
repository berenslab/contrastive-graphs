# Node Embeddings via Neighbor Embeddings

Jan Niklas Böhm\*, Marius Keute\*, Alica Guzmán, Sebastian Damrich,
Andrew Draganov, Dmitry Kobak

![alt="fig1 of the paper “Node Embeddings via Neighbor Embeddings”](https://github.com/berenslab/contrastive-graphs/releases/download/v1-data-alpha/fig1.png)

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

## Abstract

> Graph layouts and node embeddings are two distinct paradigms for
> non-parametric graph representation learning. In the former, nodes
> are embedded into 2D space for visualization purposes. In the
> latter, nodes are embedded into a high-dimensional vector space for
> downstream processing. State-of-the-art algorithms for these two
> paradigms, force-directed layouts and random-walk-based contrastive
> learning (such as DeepWalk and node2vec), have little in common. In
> this work, we show that both paradigms can be approached with a
> single coherent framework based on established neighbor embedding
> methods. Specifically, we introduce graph t-SNE, a neighbor
> embedding method for two-dimensional graph layouts, and graph CNE, a
> contrastive neighbor embedding method that produces high-dimensional
> node representations by optimizing the InfoNCE objective. We show
> that both graph t-SNE and graph CNE strongly outperform
> state-of-the-art algorithms in terms of local structure
> preservation, while being conceptually simpler.


## Code Structure

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
hold any of these files (except for the source code).  You can find
the plots and experimental data in the [releases
section](https://github.com/berenslab/contrastive-graphs/releases) of
this repository.

## Running the experiments

The experiments are launched with the script `launch.py` (in
`src/nik_graphs/`).  It takes two arguments, `--path` and `--outfile`.
The path is used to dynamically load the correct module and dispatch
it with the correct arguments.  Everything that the model outputs is
then saved to the file specified by `--outfile`.  As an example, you
could call
```sh
python3 src/nik_graphs/launch.py runs/mnist/tsne --outfile runs/mnist/tsne/1.zip
```
and that would then go on to import `tsne.py` (from
`src/nik_graphs/modules/`) and call the method
`run_path(path, outfile)`.
This works for all modules, but it does expect that every folder that
is above the current one (in the example above `runs/mnist/`) have
been run before already and the result stored in a file named `1.zip`.
So to run the example above, you should actually run
```sh
python3 src/nik_graphs/launch.py runs/mnist --outfile runs/mnist/1.zip
python3 src/nik_graphs/launch.py runs/mnist/tsne --outfile runs/mnist/tsne/1.zip
```

To automate all of this, the code uses `redo` for running the
experiments.  The program `redo`is a niche build system that is abused
here to run the experiments as well as figure out which experiments
need to be re-run in order to get up-to-date figures.  This means that
the structure of the code is somewhat rigid and has well-defined
outputs.  It resolved the dependencies of a script file and knows how
to launch the python code properly within a container.  If you're
comfortable with `sh` scripts, you can take a look at
`runs/default.zip.do` and at the do scripts in `media/` and
`dataframes/`.  They follow a similar principle in that they use a
small python script, similar to `launch.py`, to dynamically import a
python module which will then emit the dependencies and can also be
run to transform the input files into something that is processed
further.

In a way you could think of this repository as a build system that
runs incredibly long compilations in order to finally produce some
`.pdf` files as well as `.tex` files, which have then been included in
the paper.  Sounds a bit weird, but it works for me.
