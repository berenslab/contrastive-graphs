redo-ifchange $2
PROJROOT=$(dirname $PWD)
SINGULARITYFLAGS="--pwd $PWD --bind $PROJROOT --env PYTHONPATH=$PROJROOT"
RUN="singularity exec $SINGULARITYFLAGS ../nik.sif python3"
$RUN telegram_send.py $2
