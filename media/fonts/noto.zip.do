set -e
curl -sL https://github.com/notofonts/latin-greek-cyrillic/releases/download/NotoSans-v2.015/NotoSans-v2.015.zip > $3
TMP=$(mktemp)
echo "0c34df072a3fa7efbb7cbf34950e1f971a4447cffe365d3a359e2d4089b958f5 $3" \
     > $TMP
sha256sum --quiet -c $TMP
rm $TMP
