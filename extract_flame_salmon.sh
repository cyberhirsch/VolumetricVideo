#!/usr/bin/env bash
# Recombine the multi-volume zip, extract MP4s + poses_bounds.npy,
# clean up the archives.
set -e

ENV="$HOME/miniconda3/envs/lvv"
DEST="$HOME/lvv-data/neural3dv"
cd "$DEST"

echo "=== recombining multi-volume zip (zip -FF) ==="
"$ENV/bin/zip" -FF flame_salmon_1_split.zip --out flame_salmon_1.zip 2>&1 | tail -6

echo
echo "=== unzipping ==="
"$ENV/bin/unzip" -q -o flame_salmon_1.zip

echo
echo "=== extracted tree ==="
find flame_salmon_1 -maxdepth 2 | head -30
echo
du -sh flame_salmon_1
echo
echo "=== MP4 sample ==="
ls -lh flame_salmon_1/*.mp4 | head -5

echo
echo "=== cleanup zips to recover ~10 GB ==="
rm -f flame_salmon_1.zip flame_salmon_1_split.z01 flame_salmon_1_split.z02 \
      flame_salmon_1_split.z03 flame_salmon_1_split.zip
ls -lh "$DEST"
echo
df -h ~ | tail -1
echo "=== EXTRACT (mp4s) DONE ==="
