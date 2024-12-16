set -e
URL=https://github.com/be5invis/Iosevka/releases/download/v32.2.1/PkgTTF-Iosevka-32.2.1.zip
curl -sL "$URL" > $3
TMP=$(mktemp)
echo "e34417198b3d3827a7496e53929ccae96d119bab2f23a23ba1bfc5f5e037179d $3" \
     > $TMP
sha256sum --quiet -c $TMP
rm $TMP
