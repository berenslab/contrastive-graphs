set -e
curl -sL https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip > $3
TMP=$(mktemp)
echo "9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e $3" \
     > $TMP
sha256sum --quiet -c $TMP
rm $TMP
