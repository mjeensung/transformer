#!/bin/bash
# Most codes used here are referred from below:
# https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh
#
SRC=fr
TGT=en

ROOT="./datasets"
ORIG=$ROOT/iwslt17_fr.en.orig
DATA=$ROOT/iwslt17.fr.en
mkdir -p "$ORIG" "$DATA"

URL="https://wit3.fbk.eu/archive/2017-01-trnted/texts/fr/en/fr-en.tgz"
ARCHIVE=$ORIG/fr-en.tgz
VALID_SET="IWSLT17.TED.dev2010.fr-en"

# download and extract data
if [ -f "$ARCHIVE" ]; then
    echo "$ARCHIVE already exists, skipping download"
else
    wget -P "$ORIG" "$URL"
    if [ -f "$ARCHIVE" ]; then
        echo "$URL successfully downloaded."
    else
        echo "$URL not successfully downloaded."
        exit 1
    fi
fi
FILE=${ARCHIVE: -4}
if [ -e "$FILE" ]; then
    echo "$FILE already exists, skipping extraction"
else
    tar -C "$ORIG" -xzvf "$ARCHIVE"
fi

echo "pre-processing train data..."
for LANG in "${SRC}" "${TGT}"; do
    cat "$ORIG/${SRC}-${TGT}/train.tags.${SRC}-${TGT}.${LANG}" \
        | grep -v '<url>' \
        | grep -v '<talkid>' \
        | grep -v '<keywords>' \
        | grep -v '<speaker>' \
        | grep -v '<reviewer' \
        | grep -v '<translator' \
        | grep -v '<doc' \
        | grep -v '</doc>' \
        | sed -e 's/<title>//g' \
        | sed -e 's/<\/title>//g' \
        | sed -e 's/<description>//g' \
        | sed -e 's/<\/description>//g' \
        | sed 's/^\s*//g' \
        | sed 's/\s*$//g' \
        > "$DATA/train.${SRC}-${TGT}.${LANG}"
done

echo "pre-processing valid data..."
for LANG in "$SRC" "$TGT"; do
    grep '<seg id' "$ORIG/${SRC}-${TGT}/${VALID_SET}.${LANG}.xml" \
        | sed -e 's/<seg id="[0-9]*">\s*//g' \
        | sed -e 's/\s*<\/seg>\s*//g' \
        | sed -e "s/\’/\'/g" \
        > "$DATA/valid.${SRC}-${TGT}.${LANG}"
done