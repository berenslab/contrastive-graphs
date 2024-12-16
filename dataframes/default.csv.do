# -*- mode: sh -*-
uv run python ../src/nik_graphs/agg.py --dispatch $2 --printdeps | xargs redo-ifchange
uv run python ../src/nik_graphs/agg.py --dispatch $2 --outfile $3
