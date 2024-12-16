# -*- mode: sh -*-

redo-ifchange ../../iosevka.zip

# https://stackoverflow.com/questions/7148604/
# extract-list-of-file-names-in-a-zip-archive-when-unzip-l#13619930
unzip -Z -1 ../../iosevka.zip > $3
