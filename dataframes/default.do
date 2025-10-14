# -*- mode: sh -*-

# non-greedily remove eveything from the right until the first dot.
# Basically remove the extension of the file name
NOEXT="${2%.*}"

PROJROOT=$(dirname $PWD)
SINGULARITYFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR --env PYTHONPATH=$PROJROOT/src"
RUN="singularity exec $SINGULARITYFLAGS ../nik.sif python3"
$RUN ../src/nik_graphs/agg.py --dispatch $NOEXT --printdeps | xargs redo-ifchange
$RUN ../src/nik_graphs/agg.py --dispatch $NOEXT --outfile $3
