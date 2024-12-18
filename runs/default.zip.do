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

PYNAME=$(rstrip $(basename $_PATH) ",*")
PARTITION=$(grep "__partition__ = " ../src/nik_graphs/modules/${PYNAME}.py \
                | sed 's/__partition__ = "\(.*\)"/\1/')
FLAGS=--cpus-per-task=8

if [ ! -f ../nik.sif ]; then
    echo "$0: Container file 'nik.sif' does not exist. Build it first!" >&2
    exit 1
fi

PROJROOT=$(dirname $PWD)
# those directories will be available inside the container, see
# https://docs.sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html#user-defined-bind-paths
SINGULARITY_BINDPATH="$PROJROOT,$XDG_CACHE_DIR"
SINGULARITYFLAGS="--pwd $PWD --env PYTHONPATH=$PROJROOT/src"
RUN="singularity exec $SINGULARITYFLAGS ../nik.sif python3"
# The actual calls to the computation happen in the block below.  We
# first determine whether the current file needs to be launched on a
# partition (and if we can even do that) and then pass the flags to srun.
# The call chain is [srun ->] singularity -> python3 launch.py

# if $PARTITION is not set or if `srun` does not exist, we call `uv` directly
if [ x$PARTITION == x -o x$(command -v srun) == x ]; then
    $RUN ../src/nik_graphs/launch.py --path $_PATH --outfile $3
elif [ x$PARTITION == "xcpu-galvani" ]; then
    srun --partition $PARTITION $FLAGS \
         $RUN ../src/nik_graphs/launch.py --path $_PATH --outfile $3
# if PARTITION is a GPU partition, we also need to pass the flag for GPU
elif [ x$PARTITION == "x2080-galvani"  -o x$PARTITION == "xa100-galvani" ]; then
    srun --partition $PARTITION --gpus=1 $FLAGS \
         $RUN ../src/nik_graphs/launch.py --path $_PATH --outfile $3
else
    echo "$0: Unknown partition \"$PARTITION\" found in $_PATH" >&2
    exit 1
fi

if [ -f $_PATH/files.dep ]; then
    xargs redo-ifchange < $_PATH/files.dep
else
    redo-ifcreate $_PATH/files.dep
fi
