# -*- mode: sh -*-

_PATH=$(dirname $(realpath $2))
PARENT=$(dirname $_PATH)
# echo $_PATH >&2

if [ $PWD != $PARENT ]; then
    redo-ifchange $PARENT/1.zip
fi

uv run python ../src/nik_graphs/launch.py --path $_PATH --outfile $3

if [ -f $_PATH/files.dep ]; then
    xargs redo-ifchange < $_PATH/files.dep
else
    redo-ifcreate $_PATH/files.dep
fi
