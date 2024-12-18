redo-ifchange $2
PROJROOT=$(dirname $PWD)
SINGULARITY_BINDPATH="$PROJROOT,$XDG_CACHE_DIR"
SINGULARITYFLAGS="--pwd $PWD --env PYTHONPATH=$PROJROOT/src"
RUN="singularity exec $SINGULARITYFLAGS ../nik.sif python3"
$RUN telegram_send.py $2
