#!/usr/bin/env bash
# Phase 0 setup: conda env + CUDA toolkit + compilers + CUDA PyTorch
set -e
set -o pipefail

CONDA="$HOME/miniconda3/bin/conda"
echo "=== [1/5] Accepting conda ToS ==="
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$CONDA" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo "=== [2/5] Ensuring env 'lvv' (python 3.10) exists ==="
if "$CONDA" env list | grep -q "/envs/lvv"; then
    echo "env lvv already exists"
else
    "$CONDA" create -y -n lvv python=3.10
fi

ENV="$HOME/miniconda3/envs/lvv"
PY="$ENV/bin/python"
PIP="$ENV/bin/pip"

echo "=== [3/5] Installing CUDA 12.1 toolkit + gcc/g++ 12 into env ==="
"$CONDA" install -y -n lvv -c nvidia/label/cuda-12.1.1 cuda-toolkit
"$CONDA" install -y -n lvv -c conda-forge gxx=12 gcc=12 ninja

echo "=== [4/5] Installing CUDA-enabled PyTorch (cu121) ==="
"$PIP" install --no-input torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

echo "=== [5/5] Verifying ==="
"$ENV/bin/nvcc" --version | tail -2
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda avail', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo "=== SETUP PHASE 0a COMPLETE ==="
