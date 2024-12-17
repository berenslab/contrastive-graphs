redo-ifchange $2
RUN="singularity exec ../nik.sif --exec --pwd \"$PWD\" --bind \"$PWD\" python3"
$RUN telegram_send.py $2
