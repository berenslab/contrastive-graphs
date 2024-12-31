set -e
curl -sL https://github.com/googlefonts/roboto-3-classic/releases/download/v3.010/Roboto_v3.010.zip > $3
TMP=$(mktemp)
echo "173f6d2fdd7e523f189ccf04507f72d900e88cd18332a86f4e33afb3bd78dc30 $3" \
     > $TMP
sha256sum --quiet -c $TMP
rm $TMP
