#!/usr/bin/env bash
# Phase 0 setup (retry): CUDA toolkit + compilers + CUDA PyTorch
set -e
set -o pipefail

CONDA="$HOME/miniconda3/bin/conda"
ENV="$HOME/miniconda3/envs/lvv"
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"

echo "=== Configuring conda for resilient downloads ==="
"$CONDA" config --set remote_max_retries 10
"$CONDA" config --set remote_backoff_factor 2
"$CONDA" config --set remote_connect_timeout_secs 60
"$CONDA" config --set remote_read_timeout_secs 120

echo "=== [3/5] Installing CUDA 12.1 toolkit + gcc/g++ 12 (with retries) ==="
for attempt in 1 2 3 4 5; do
    echo "--- cuda-toolkit attempt $attempt ---"
    if "$CONDA" install -y -n lvv -c nvidia/label/cuda-12.1.1 cuda-toolkit; then
        echo "cuda-toolkit OK"; break
    fi
    echo "attempt $attempt failed, retrying..."; sleep 5
done
for attempt in 1 2 3 4 5; do
    echo "--- compilers attempt $attempt ---"
    if "$CONDA" install -y -n lvv -c conda-forge gxx=12 gcc=12 ninja; then
        echo "compilers OK"; break
    fi
    echo "attempt $attempt failed, retrying..."; sleep 5
done

echo "=== [4/5] Installing CUDA-enabled PyTorch (cu121) ==="
for attempt in 1 2 3 4 5; do
    echo "--- torch attempt $attempt ---"
    if "$PIP" install --no-input torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121; then
        echo "torch OK"; break
    fi
    echo "attempt $attempt failed, retrying..."; sleep 5
done

echo "=== [5/5] Verifying ==="
"$ENV/bin/nvcc" --version | tail -2
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "=== SETUP PHASE 0a COMPLETE ==="
