redo-ifchange $2
RUN="singularity exec --pwd \"$PWD\" --bind \"$(dirname $PWD)\" ../nik.sif python3"
$RUN telegram_send.py $2
