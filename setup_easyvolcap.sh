#!/usr/bin/env bash
# Phase 0: install EasyVolcap core deps + editable install
set -o pipefail

CONDA="$HOME/miniconda3/bin/conda"
ENV="$HOME/miniconda3/envs/lvv"
PIP="$ENV/bin/pip"
PY="$ENV/bin/python"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"

echo "=== [1/3] libjpeg-turbo (for PyTurboJPEG) via conda ==="
"$CONDA" install -y -n lvv -c conda-forge libjpeg-turbo || echo "libjpeg-turbo install issue (continuing)"

echo "=== [2/3] core python deps (per-package, continue on failure) ==="
PKGS="av tqdm pdbr h5py yapf ujson PyGLM scipy sympy addict PyYaml psutil ipython \
trimesh imageio PyOpenGL pycolmap PyMCubes pyperclip pyntcloud websockets \
PyTurboJPEG tensorboard ruamel.yaml cuda-python scikit-image imgui-bundle \
opencv-python plyfile lpips"
FAILED=""
for pkg in $PKGS; do
    ok=0
    for attempt in 1 2 3 4 5; do
        if "$PIP" install --no-input --retries 10 --timeout 120 "$pkg"; then ok=1; break; fi
        echo "  retry $attempt for $pkg"; sleep 3
    done
    [ "$ok" = 1 ] && echo "OK: $pkg" || { echo "FAILED: $pkg"; FAILED="$FAILED $pkg"; }
done

echo "=== [3/3] editable install of EasyVolcap (no build isolation, no deps) ==="
"$PIP" install --no-input -e "$EVC" --no-build-isolation --no-deps || echo "editable install issue"

echo "=== verifying ==="
cd "$EVC"
"$PY" -c "import easyvolcap; print('easyvolcap import OK')" 2>&1 | tail -3
echo "FAILED PACKAGES:$FAILED"
echo "=== EASYVOLCAP SETUP DONE ==="
