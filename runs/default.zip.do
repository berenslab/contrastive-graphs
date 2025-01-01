# -*- mode: sh -*-

# shell function adapted from
# https://github.com/dylanaraps/pure-sh-bible
# ?tab=readme-ov-file#strip-pattern-from-end-of-string

# usage: rstrip_after "bla,this will,be,gone" ",*"
# gives => "bla"
rstrip() {
    # Usage: rstrip "string" "pattern"
    printf '%s\n' "${1%%$2}"
}

_PATH=$(dirname $(realpath $2))
PARENT=$(dirname $_PATH)
# echo $_PATH >&2

if [ $PWD != $PARENT ]; then
    redo-ifchange $PARENT/1.zip
fi

if [ ! -f ../nik.sif ]; then
    echo "$0: Container file 'nik.sif' does not exist. Build it first!" >&2
    exit 1
fi

PROJROOT=$(dirname $PWD)
# those directories will be available inside the container, see
# https://docs.sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html#user-defined-bind-paths
SFLAGS="--pwd $PWD --bind $PROJROOT,$XDG_CACHE_DIR --env PYTHONPATH=$PROJROOT/src"

PYNAME=$(rstrip $(basename $_PATH) ",*")

# special case, we depend on the executable being present by the time
# we redo drgraph and call the launch script.  Cannot redo it inside
# of drgraph.py because that is inside of the singularity container
# (and hence there is no redo and it would be detached from the redo
# process outside of the container).
# Analogue for sgtsnepi.py, which needs the sysimage to function.
if [ x$PYNAME == xdrgraph ]; then
    redo-ifchange ../bin/drgraph
elif [ x$PYNAME == xsgtsnepi]; then
    redo-ifchange ../bin/julia/nik_sgtsnepi_sysimage.so
fi

if [ ! -f "../src/nik_graphs/modules/${PYNAME}.py" ]; then
    echo "$0: file \"../src/nik_graphs/modules/${PYNAME}.py\" does not exist" >&2
    echo "cannot launch redo.  Make sure $PYNAME.py exists as a module" >&2
    exit 1
fi
PARTITION=$(grep "__partition__ = " ../src/nik_graphs/modules/${PYNAME}.py \
                | sed 's/__partition__ = "\(.*\)"/\1/')

# figure out which partition we are on, and set the partition
# accordingly, plus add the --nv flag to the singularity flags so that
# the GPU is visible when inside of the container.
if [ x$PARTITION == "xcpu-galvani" ]; then
    PARTITIONFLAGS="--partition $PARTITION"
elif [ x$PARTITION == "x2080-galvani"  -o x$PARTITION == "xa100-galvani" ]; then
    PARTITIONFLAGS="--partition $PARTITION --gpus=1"
    # append --nv so that cuda will work in the container
    SFLAGS="$SFLAGS --nv"
elif [ x$PARTITION == "x" ]; then
    PARTITIONFLAGS=
else
    echo "$0: Unknown partition \"$PARTITION\" found in $_PATH" >&2
    exit 1
fi
SLURMFLAGS="--cpus-per-task=8 --job-name $(basename $_PATH) $PARTITIONFLAGS"

if [ x$(command -v srun) != x  -a "x$PARTITION" != x ]; then
    SRUN="srun --quiet --partition $PARTITION $SLURMFLAGS"
else
    SRUN=
fi
RUN="singularity exec $SFLAGS ../nik.sif python3"

# The actual calls to the computation happen in the line below.
# The call chain is [srun ->] singularity -> python3 launch.py
# If $SRUN is not defined, then the variable will be empty, hence we
# will launch $RUN (the call to singularity) directly on the current
# computer.
$SRUN $RUN ../src/nik_graphs/launch.py --path $_PATH --outfile $3

if [ -f $_PATH/files.dep ]; then
    xargs redo-ifchange < $_PATH/files.dep
else
    redo-ifcreate $_PATH/files.dep
fi
