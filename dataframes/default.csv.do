# -*- mode: sh -*-

redo-ifchange "$2.parquet"

PROJROOT=$(dirname $PWD)
SINGULARITYFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR --env PYTHONPATH=$PROJROOT/src"
RUN="singularity exec $SINGULARITYFLAGS ../nik.sif python3"
$RUN -c "import polars as pl; pl.scan_parquet(\"$2.parquet\").sink_csv(\"$3\")"
