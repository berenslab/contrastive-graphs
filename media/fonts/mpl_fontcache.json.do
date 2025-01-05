# -*- mode: sh -*-

redo-ifchange ttf/*/all-fonts mpl_dump_fonts.py

PROJROOT=$(dirname $PWD)
SINGULARITYFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR"
RUN="singularity exec $SINGULARITYFLAGS ../../nik.sif python3"
$RUN mpl_dump_fonts.py $3
