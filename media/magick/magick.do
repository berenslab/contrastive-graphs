# -*- mode: sh -*-
set -e
exec >&2

MAGICK_VERSION=7.1.1-41
TMPTAR=$(mktemp)
curl -s https://imagemagick.org/archive/ImageMagick-${MAGICK_VERSION}.tar.gz > $TMPTAR
TMP=$(mktemp)
echo "81c03fe273d8dd33c36dc5b967ae279f87e3be5bb5070e6fbeb893ddd40b0340  $TMPTAR" \
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
