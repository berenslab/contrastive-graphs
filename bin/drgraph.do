# -*- mode: sh -*-
exec >&2
TMPDIR=$(mktemp --directory)
git clone https://github.com/ZJUVAI/DRGraph $TMPDIR
OLDPWD=$PWD
cd $TMPDIR
mkdir build
sh build.sh
cd $OLDPWD
mv $TMPDIR/Vis $3
rm -rf $TMPDIR
