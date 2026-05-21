#!/usr/bin/env bash
# Phase 1: download EasyVolcap example dataset (ENeRF-Outdoor actor1_4_subseq)
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
PIP="$ENV/bin/pip"
PY="$ENV/bin/python"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
DATA="$HOME/lvv-data"           # fast WSL-native storage for training I/O
FILEID="1XxeO7TnAPvDugnxguEF5Jp89ERS9CAia"

mkdir -p "$DATA"

echo "=== [1/4] installing gdown ==="
"$PIP" install --no-input --retries 10 gdown

echo "=== [2/4] downloading example dataset from Google Drive ==="
cd "$DATA"
for attempt in 1 2 3 4 5; do
    if "$ENV/bin/gdown" --id "$FILEID" -O example_dataset.zip; then echo "download OK"; break; fi
    echo "retry $attempt"; sleep 5
done
ls -lh example_dataset.zip

echo "=== [3/4] extracting ==="
"$PY" -c "import zipfile; zipfile.ZipFile('example_dataset.zip').extractall('.')"
echo "--- extracted tree (top 2 levels) ---"
find "$DATA" -maxdepth 3 -not -path '*/\.*' | head -40

echo "=== [4/4] symlinking EasyVolcap/data -> $DATA ==="
if [ ! -e "$EVC/data" ]; then ln -s "$DATA" "$EVC/data"; echo "symlink created"; else echo "data path already exists: $EVC/data"; fi
ls -la "$EVC/data" 2>&1 | head -5
echo "=== DATASET SETUP DONE ==="
