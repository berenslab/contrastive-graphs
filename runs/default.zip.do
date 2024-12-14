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

# if $PARTITION is not set or if `srun` does not exist, we call `uv` directly
if [ x$PARTITION == x -o x$(command -v srun) == x ]; then
    uv run python ../src/nik_graphs/launch.py --path $_PATH --outfile $3
elif [ x$PARTITION == "xcpu-galvani" ]; then
    srun --partition $PARTITION \
         uv run python ../src/nik_graphs/launch.py --path $_PATH --outfile $3
# if PARTITION is a GPU partition, we also need to pass the flag for GPU
elif [ x$PARTITION == "x2080-galvani" ]; then
    srun --partition $PARTITION --gres gpu:1 \
         uv run python ../src/nik_graphs/launch.py --path $_PATH --outfile $3
else
    echo "Unknown partition \"$PARTITION\" found in $_PATH" >&2
    exit 1
fi

if [ -f $_PATH/files.dep ]; then
    xargs redo-ifchange < $_PATH/files.dep
else
    redo-ifcreate $_PATH/files.dep
fi
