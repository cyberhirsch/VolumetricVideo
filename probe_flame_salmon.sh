#!/usr/bin/env bash
set -e

ENV="$HOME/miniconda3/envs/lvv"
SRC="$HOME/lvv-data/neural3dv/flame_salmon_1"

echo "=== all cam files ==="
ls -1 "$SRC"/cam*.mp4 | sort
echo "count: $(ls -1 "$SRC"/cam*.mp4 | wc -l)"

echo
echo "=== cam00 metadata ==="
"$ENV/bin/ffprobe" -v error -select_streams v:0 \
    -show_entries stream=width,height,r_frame_rate,nb_frames,codec_name,duration \
    -of default "$SRC/cam00.mp4"

echo
echo "=== poses_bounds.npy ==="
"$ENV/bin/python" - <<'PY'
import os, numpy as np
p = np.load(os.path.expanduser("~/lvv-data/neural3dv/flame_salmon_1/poses_bounds.npy"))
print("shape:", p.shape, "dtype:", p.dtype)
print("first row:", p[0])
print("near/far range:", p[:, 15].min(), "..", p[:, 15].max(), "/", p[:, 16].min(), "..", p[:, 16].max())
# LLFF format: 17 = [3x5 pose] flattened (15) + 2 (near, far)
pose0 = p[0, :15].reshape(3, 5)
print("cam0 pose 3x5:")
print(pose0)
print("hwf:", pose0[:, 4])
PY
