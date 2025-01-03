# -*- mode: sh -*-

set -e
TMPTAR=$(mktemp)
TMPSHA=$(mktemp)
GSVERSION=10.04.0
curl -sL https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/gs10040/ghostscript-${GSVERSION}.tar.gz \
     > $TMPTAR
echo "c764dfbb7b13fc71a7a05c634e014f9bb1fb83b899fe39efc0b6c3522a9998b1 $TMPTAR" \
     > $TMPSHA
sha256sum --quiet -c $TMPSHA || exit 55

tar xf $TMPTAR
rm $TMPTAR $TMPSHA
BINDIR=$PWD
cd ghostscript-${GSVERSION}
./configure
make -j 16
cd $BINDIR
mv ghostscript-${GSVERSION}/bin/gs $3
rm -r ghostscript-${GSVERSION}
