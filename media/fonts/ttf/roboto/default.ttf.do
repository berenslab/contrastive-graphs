# -*- mode: sh -*-

redo-ifchange ../../roboto.zip
ARCHIVENAME=$(unzip -Z -1 ../../roboto.zip \
                    | grep "^Roboto_v.*/hinted/static/$2\.ttf")

unzip -p ../../roboto.zip "$ARCHIVENAME" > $3
