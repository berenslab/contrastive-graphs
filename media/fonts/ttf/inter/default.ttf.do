# -*- mode: sh -*-

redo-ifchange ../../inter.zip
unzip -p ../../inter.zip extras/ttf/$2.ttf > $3
