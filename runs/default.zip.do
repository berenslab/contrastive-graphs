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

# The actual calls to the computation happen in the block below.  We
# first determine whether the current file needs to be launched on a
# partition (and if we can even do that) and then pass the flags to srun.
# The call chain is [srun ->] uv -> python launch.py

# if $PARTITION is not set or if `srun` does not exist, we call `uv` directly
if [ x$PARTITION == x -o ! $(command -v srun) ]; then
    uv run python ../src/nik_graphs/launch.py --path $_PATH --outfile $3
elif [ $PARTITION == "cpu-galvani" ]; then
    srun --partition $PARTITION $FLAGS \
         uv run python ../src/nik_graphs/launch.py --path $_PATH --outfile $3
# if PARTITION is a GPU partition, we also need to pass the flag for GPU
elif [ $PARTITION == "2080-galvani" ]; then
    srun --partition $PARTITION --gpus=1 $FLAGS \
         uv run python ../src/nik_graphs/launch.py --path $_PATH --outfile $3
else
    echo "Unknown partition" $PARTITION "found in" $_PATH >&2
    exit 1
fi

if [ -f $_PATH/files.dep ]; then
    xargs redo-ifchange < $_PATH/files.dep
else
    redo-ifcreate $_PATH/files.dep
fi
