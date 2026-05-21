#!/usr/bin/env bash
# Resumable download of Neural3DV flame_salmon (4 split-zip parts, ~5 GB total)
# Goes to WSL native fs; combine + extract is a SEPARATE later step.
set -o pipefail

DEST="$HOME/lvv-data/neural3dv"
mkdir -p "$DEST"
cd "$DEST"

BASE="https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0"
FILES=(
    "flame_salmon_1_split.z01"   # 1.57 GB
    "flame_salmon_1_split.z02"   # 1.57 GB
    "flame_salmon_1_split.z03"   # 1.57 GB
    "flame_salmon_1_split.zip"   # 273 MB
)

echo "=== destination: $DEST ==="
df -h "$DEST" | tail -1

for f in "${FILES[@]}"; do
    echo
    echo "=== $f ==="
    if [ -f "$f" ]; then
        echo "already present: $(ls -lh "$f" | awk '{print $5}')"
    fi
    # --continue resumes partial files; ample retries for the flaky-TLS network
    wget --continue --tries=100 --waitretry=10 --timeout=120 --read-timeout=120 \
         --retry-connrefused --no-verbose --show-progress \
         -O "$f" "$BASE/$f"
    echo "got: $(ls -lh "$f" | awk '{print $5}')  ($f)"
done

echo
echo "=== final listing ==="
ls -lh "$DEST"
echo
echo "=== total ==="
du -sh "$DEST"
echo "=== free now ==="
df -h "$DEST" | tail -1
echo "=== FLAME_SALMON DOWNLOAD DONE ==="
