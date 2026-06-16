#!/usr/bin/env bash
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/visual_genome"
mkdir -p "$DEST"
cd "$DEST"

echo "==> Downloading metadata..."
wget -c https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
wget -c https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip
wget -c https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip

echo "==> Downloading images (~15GB)..."
aria2c -x 16 -s 16 -c -o images.zip  https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
aria2c -x 16 -s 16 -c -o images2.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

echo "==> Extracting metadata..."
unzip -o image_data.json.zip       && rm image_data.json.zip
unzip -o relationships.json.zip    && rm relationships.json.zip
unzip -o question_answers.json.zip && rm question_answers.json.zip

echo "==> Extracting images..."
unzip -o images.zip  && rm images.zip
unzip -o images2.zip && rm images2.zip

echo "==> Done. Contents of $DEST:"
ls -lh "$DEST"
