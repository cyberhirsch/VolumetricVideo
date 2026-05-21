#!/usr/bin/env bash
# Phase 0: install torch's NVIDIA CUDA runtime deps (from PyPI) with retries
set -e
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
PIP="$ENV/bin/pip"

DEPS="nvidia-cublas-cu12==12.1.3.1 \
nvidia-cuda-cupti-cu12==12.1.105 \
nvidia-cuda-nvrtc-cu12==12.1.105 \
nvidia-cuda-runtime-cu12==12.1.105 \
nvidia-cudnn-cu12==9.1.0.70 \
nvidia-cufft-cu12==11.0.2.54 \
nvidia-curand-cu12==10.3.2.106 \
nvidia-cusolver-cu12==11.4.5.107 \
nvidia-cusparse-cu12==12.1.0.106 \
nvidia-nccl-cu12==2.20.5 \
nvidia-nvtx-cu12==12.1.105 \
triton==3.0.0"

echo "=== installing torch CUDA runtime deps (with retries) ==="
for attempt in 1 2 3 4 5 6 7 8; do
    echo "--- attempt $attempt ---"
    if "$PIP" install --no-input --retries 10 --timeout 120 $DEPS; then
        echo "deps OK"; break
    fi
    echo "attempt $attempt failed, retrying..."; sleep 5
done

echo "=== verifying ==="
"$ENV/bin/python" -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"
"$ENV/bin/python" -c "import torch; x=torch.randn(1000,1000,device='cuda'); print('matmul on GPU OK:', (x@x).sum().item() != 0)"
echo "=== TORCH FULLY WORKING ==="
