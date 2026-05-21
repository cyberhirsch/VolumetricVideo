#!/usr/bin/env bash
# Phase 0 addendum: build tiny-cuda-nn bindings
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
PIP="$ENV/bin/pip"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export TCNN_CUDA_ARCHITECTURES="86"
export MAX_JOBS=8
# WSL: libcuda.so (CUDA driver) lives here; conda stubs as fallback for the linker
export LIBRARY_PATH="$ENV/lib/stubs:/usr/lib/wsl/lib:$LIBRARY_PATH"

echo "=== building tinycudann (RTX 3090, sm_86) ==="
for attempt in 1 2 3 4 5; do
    echo "--- attempt $attempt ---"
    if "$PIP" install --no-input --no-build-isolation \
        "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"; then
        echo "tinycudann OK"; break
    fi
    echo "attempt $attempt failed, retrying..."; sleep 5
done

echo "=== verifying ==="
"$ENV/bin/python" -c "import torch; import tinycudann as tcnn; print('tinycudann OK')" 2>&1 | tail -3
echo "=== TCNN SETUP DONE ==="
