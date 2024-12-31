# -*- mode: sh -*-

redo-ifchange ../../roboto.zip

# https://stackoverflow.com/questions/7148604/
# extract-list-of-file-names-in-a-zip-archive-when-unzip-l#13619930
unzip -Z -1 ../../roboto.zip \
    | grep '^Roboto_v.*/hinted/static/.*\.ttf' | sed 's#.*/##' > $3
xargs redo-ifchange < $3
