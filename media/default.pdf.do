# -*- mode: sh -*-
PROJROOT=$(dirname $PWD)
SINGULARITYFLAGS="--pwd $PWD --bind $PROJROOT --env PYTHONPATH=$PROJROOT/src"
RUN="singularity exec $SINGULARITYFLAGS ../nik.sif python3"
$RUN ../src/nik_graphs/plot.py --plotname $2 --printdeps | xargs redo-ifchange
$RUN ../src/nik_graphs/plot.py --plotname $2 --outfile $3
