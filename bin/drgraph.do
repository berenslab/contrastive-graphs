# -*- mode: sh -*-
exec >&2
TMPDIR=$(mktemp --directory)
git clone https://github.com/ZJUVAI/DRGraph $TMPDIR
OLDPWD=$PWD
cd $TMPDIR
mkdir build
singularity exec --bind $TMPDIR --pwd $TMPDIR $OLDPWD/../nik.sif sh build.sh
cd $OLDPWD
mv $TMPDIR/Vis $3
rm -rf $TMPDIR
