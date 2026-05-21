#!/usr/bin/env bash
# Phase 0: build CUDA extensions (--no-build-isolation so they see env torch)
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
PIP="$ENV/bin/pip"
PY="$ENV/bin/python"

export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"   # RTX 3090 = Ampere sm_86
export MAX_JOBS=8

echo "=== CUDA_HOME=$CUDA_HOME ==="
"$ENV/bin/nvcc" --version | tail -2

retry_pip() {
    for attempt in 1 2 3 4 5 6 7 8; do
        echo "--- pip attempt $attempt: $* ---"
        if "$PIP" install --no-input --retries 10 --timeout 120 "$@"; then
            echo "OK: $*"; return 0
        fi
        echo "attempt $attempt failed, retrying..."; sleep 5
    done
    echo "FAILED after retries: $*"; return 1
}

echo "=== [1/4] Build tooling ==="
retry_pip ninja setuptools wheel

echo "=== [2/4] simple_knn (CUDA build, no-build-isolation) ==="
retry_pip --no-build-isolation "git+https://gitlab.inria.fr/bkerbl/simple-knn.git" || true

echo "=== [3/4] diff_gauss (CUDA build, dendenxu fork, no-build-isolation) ==="
retry_pip --no-build-isolation "git+https://github.com/dendenxu/diff-gaussian-rasterization" || true

echo "=== [4/4] fast_gauss (shader-based, no CUDA compile) ==="
retry_pip fast_gauss || true

echo "=== verifying extension imports ==="
"$PY" -c "import simple_knn._C as m; print('simple_knn OK')" 2>&1 | tail -1
"$PY" -c "import diff_gauss; print('diff_gauss OK')" 2>&1 | tail -1
"$PY" -c "import fast_gauss; print('fast_gauss OK')" 2>&1 | tail -1
echo "=== EXTENSIONS SCRIPT DONE ==="
