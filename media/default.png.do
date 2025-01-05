# -*- mode: sh -*-

redo-ifchange "$2.pdf"

if [ $(command -v pdftoppm) ]; then
    # leaving out a root name (second arg) when using `pdftoppm`
    # apparently writes the output to stdout.  This is not documented
    # anywhere in the man page as far as I can see.
    pdftoppm -r 300 -png -singlefile "$2.pdf" > $3
    redo-ifcreate ../bin/magick
elif [ $(command -v convert) ]; then
    convert -density 600 "$2.pdf" -resize 2000x2000 png:- > $3 2>/dev/null
    redo-ifcreate ../bin/magick
else
    redo-ifchange ../bin/magick
    if [ $(command -v gs) ]; then
        redo-ifcreate ../bin/gs
    else
        redo-ifchange ../bin/gs
    fi

    # add bin/ to PATH because the executable `gs` is located in there
    # (in case it is not found on the system).
    PATH=$PATH:$PWD/../bin ../bin/magick convert -density 600 "$2.pdf" -resize 50% png:- > $3 2>/dev/null
fi
