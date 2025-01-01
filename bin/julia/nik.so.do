# -*- mode: sh -*-
exec >&2

redo-ifchange make_sysimage.jl precompile_sgtsnepi.jl
PROJROOT=$(dirname $(dirname $PWD))
SFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR --env JULIADEPOT=$PWD --env SSL_CERT_FILE="
singularity exec $SFLAGS ../../nik.sif /opt/julia-1.11.2/bin/julia make_sysimage.jl $3
