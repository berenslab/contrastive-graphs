# -*- mode: sh -*-

# source this script from redo (it reuses $1, $2, $3).  The format
# extension needs to be specified as $TBLEXT (parquet, tex, txt, ...).
set -e
PROJROOT=$(dirname $(dirname $PWD))
SINGULARITYFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR --env PYTHONPATH=$PROJROOT/src"
RUN="singularity exec $SINGULARITYFLAGS ../../nik.sif python3"
$RUN ../../src/nik_graphs/tables/__init__.py --dispatch $2 --printdeps --format $TBLEXT | xargs redo-ifchange
$RUN ../../src/nik_graphs/tables/__init__.py --dispatch $2 --outfile $3 --format $TBLEXT
