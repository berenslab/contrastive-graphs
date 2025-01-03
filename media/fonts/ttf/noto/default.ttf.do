# -*- mode: sh -*-

redo-ifchange ../../noto.zip
unzip -p ../../noto.zip NotoSans/full/ttf/$2.ttf > $3
