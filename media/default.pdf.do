# -*- mode: sh -*-
RUN="singularity exec ../nik.sif --exec --pwd \"$PWD\" --bind \"$PWD\" python3"
$RUN ../src/nik_graphs/plot.py --plotname $2 --printdeps | xargs redo-ifchange
$RUN ../src/nik_graphs/plot.py --plotname $2 --outfile $3
