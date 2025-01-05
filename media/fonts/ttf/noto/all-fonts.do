# -*- mode: sh -*-

redo-ifchange ../../noto.zip

# https://stackoverflow.com/questions/7148604/
# extract-list-of-file-names-in-a-zip-archive-when-unzip-l#13619930
unzip -Z -1 ../../noto.zip \
    | grep --invert-match -E "Semi|Extra" \
    | grep '^NotoSans/full/ttf/.*\.ttf' | sed 's#.*/##' > $3
xargs redo-ifchange < $3
