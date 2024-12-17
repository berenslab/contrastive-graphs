# -*- mode: sh -*-
RUN="singularity exec ../nik.sif --exec --pwd \"$PWD\" --bind \"$(dirname $PWD)\" python3"
$RUN ../src/nik_graphs/agg.py --dispatch $2 --printdeps | xargs redo-ifchange
$RUN ../src/nik_graphs/agg.py --dispatch $2 --outfile $3
