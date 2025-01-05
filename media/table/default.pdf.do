# -*- mode: sh -*-
exec >&2
set -e


# adapted from https://redo.readthedocs.io/en/latest/cookbook/latex/
redo-ifchange $2.tex
TMPDIR=$(mktemp --directory)
LATEX=lualatex

touch "$TMPDIR/$2.aux.old"
OK=1

for i in 0 1 2 3 4 5; do

    $LATEX --halt-on-error \
        --output-directory="$TMPDIR" \
        --recorder \
        "$2.tex" < /dev/null >/dev/null 2> /dev/null

    if diff "$TMPDIR/$2.aux.old" \
            "$TMPDIR/$2.aux" > /dev/null; then
        # .aux file converged, so we're done
        OK=0
        break
    fi
    # echo
    # echo "$0: $2.aux changed: try again (try #$i)"
    # echo
    cp "$TMPDIR/$2.aux" "$TMPDIR/$2.aux.old"
done

if [ "$OK" -eq "1" ]; then
    echo "$0: fatal: $2.aux did not converge!" >&2
    exit 10
fi

# With --recorder, latex produces a list of files
# it used during its run.  Let's depend on all of
# them, so if they ever change, we'll redo.
# grep ^INPUT "$TMPDIR/$2.fls" |
#     cut -d' ' -f2 |
#     xargs redo-ifchange

mv "$TMPDIR/$2.pdf" $3
rm -r "$TMPDIR"
