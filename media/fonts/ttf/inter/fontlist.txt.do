# -*- mode: sh -*-

redo-ifchange ../../inter.zip

# https://stackoverflow.com/questions/7148604/
# extract-list-of-file-names-in-a-zip-archive-when-unzip-l#13619930
unzip -Z -1 ../../inter.zip \
    | grep 'extras/ttf/Inter' | sed 's#extras/ttf/##' > $3
