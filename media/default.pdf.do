# -*- mode: sh -*-
uv run python ../src/nik_graphs/plot.py --plotname $2 --printdeps | xargs redo-ifchange
uv run python ../src/nik_graphs/plot.py --plotname $2 --outfile $3
