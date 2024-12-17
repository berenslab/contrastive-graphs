# -*- mode: sh -*-
redo-ifchange ../.RUN
RUN=$(<../.RUN)
$RUN ../src/nik_graphs/plot.py --plotname $2 --printdeps | xargs redo-ifchange
$RUN ../src/nik_graphs/plot.py --plotname $2 --outfile $3
