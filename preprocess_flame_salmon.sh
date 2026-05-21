#!/usr/bin/env bash
# Reorganize flame_salmon into the layout EasyVolcap's neural3dv_to_easyvolcap.py
# expects, then run that converter to extract frames + write intri/extri YAMLs.
set -e
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
N3DV_ROOT="$HOME/lvv-data/neural3dv"
SRC="$N3DV_ROOT/flame_salmon_1"
DST="$N3DV_ROOT/flame_salmon"

# ffmpeg must be on PATH because the EasyVolcap script invokes it via subprocess
export PATH="$ENV/bin:/usr/bin:/bin"

echo "=== reorganizing layout (cam*.mp4 -> flame_salmon/videos/) ==="
mkdir -p "$DST/videos"
mv "$SRC"/cam*.mp4 "$DST/videos/"
mv "$SRC/poses_bounds.npy" "$DST/poses_bounds.npy"
rmdir "$SRC"
echo "videos: $(ls $DST/videos | wc -l)"
ls "$DST/videos" | head -5
ls -la "$DST/poses_bounds.npy"

echo
echo "=== running EasyVolcap's neural3dv_to_easyvolcap.py (frames + cameras) ==="
cd "$EVC"
"$ENV/bin/python" scripts/preprocess/neural3dv_to_easyvolcap.py \
    --neural3dv_root "$N3DV_ROOT" \
    --easyvolcap_root "$N3DV_ROOT" \
    --only flame_salmon 2>&1 | tail -25

echo
echo "=== result tree ==="
find "$DST" -maxdepth 2 -not -name "*.jpg" | head -30
echo
echo "=== frame counts per cam ==="
for d in "$DST"/images/*/; do
    n=$(ls "$d" | wc -l)
    echo "  $(basename $d): $n"
done
echo
echo "=== camera YAMLs ==="
ls -la "$DST"/{intri,extri}.yml 2>&1
echo
du -sh "$DST"
echo
df -h ~ | tail -1
echo "=== PREPROCESS DONE ==="
