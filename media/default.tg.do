redo-ifchange $2
RUN="singularity exec ../nik.sif --pwd \"$PWD\" --bind \"$(dirname $PWD)\" python3"
$RUN telegram_send.py $2
