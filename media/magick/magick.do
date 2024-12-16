# -*- mode: sh -*-
set -e
exec >&2

MAGICK_VERSION=7.1.1-41
TMPTAR=$(mktemp)
curl -s https://imagemagick.org/archive/ImageMagick-${MAGICK_VERSION}.tar.gz > $TMPTAR
TMP=$(mktemp)
echo "601de0e422758f5170b803d4ef02d06ec6a27addde87cf67d1ff3a21cdc9cf5d  $TMPTAR" \
     > $TMP

sha256sum --quiet -c $TMP || exit 1
rm $TMP

tar xf $TMPTAR
rm $TMPTAR
cd ImageMagick-${MAGICK_VERSION}
./configure --disable-installed
make -j 16
cd ..
ln -s ImageMagick-${MAGICK_VERSION}/utilities/magick $3
