# -*- mode: sh -*-
redo-ifchange ../.RUN
RUN=$(<../.RUN)
$RUN ../src/nik_graphs/agg.py --dispatch $2 --printdeps | xargs redo-ifchange
$RUN ../src/nik_graphs/agg.py --dispatch $2 --outfile $3
