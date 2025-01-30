# -*- mode: sh -*-

TMP=$(mktemp)
curl -sL install-tl-unx.tar.gz https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz \
    | gunzip -c | tar xf -

OLDPWD=$PWD
rm -rf tex
mv install-tl-* tex
# TLDIR=$(ls -d install-tl-*)
cd tex
perl install-tl \
     --no-interaction \
     --texdir $PWD/texlive/2025 \
     --texuserdir $PWD/texmf \
     --no-doc-install \
     --no-src-install > /dev/null

touch $OLDPWD/$3

# ./tex/texlive/2025/bin/x86_64-linux/
