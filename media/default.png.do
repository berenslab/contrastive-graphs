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
    ../bin/magick convert -density 600 "$2.pdf" -resize 2000x2000 png:- > $3 2>/dev/null
fi
