# -*- mode: sh -*-
exec >&2
SFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR --env JULIADEPOT=$PWD"
singularity exec $SFLAGS ../../nik.sif /opt/julia-1.11.2/bin/julia make_sysimage.jl $3
