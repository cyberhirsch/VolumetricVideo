#!/usr/bin/env bash
# Phase 0: resilient PyTorch download via wget --continue, then local install
set -e
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
PIP="$ENV/bin/pip"
DL="$HOME/wheels"
mkdir -p "$DL"

BASE="https://download.pytorch.org/whl/cu121"
TORCH="torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
TV="torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
TA="torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

dl() {
    local fname="$1"
    local out="$DL/${fname//%2B/+}"
    echo "=== downloading $fname ==="
    wget --continue --tries=50 --waitretry=5 --timeout=60 --read-timeout=60 \
         --retry-connrefused --no-verbose \
         -O "$out" "$BASE/$fname"
    echo "got: $(ls -lh "$out" | awk '{print $5, $9}')"
}

dl "$TORCH"
dl "$TV"
dl "$TA"

echo "=== installing wheels locally ==="
"$PIP" install --no-input --no-deps "$DL/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl" \
                                    "$DL/torchvision-0.19.1+cu121-cp310-cp310-linux_x86_64.whl" \
                                    "$DL/torchaudio-2.4.1+cu121-cp310-cp310-linux_x86_64.whl"
echo "=== installing torch runtime deps ==="
"$PIP" install --no-input filelock typing-extensions sympy networkx jinja2 fsspec numpy pillow

echo "=== verifying ==="
"$ENV/bin/python" -c "import torch, torchvision; print('torch', torch.__version__, 'tv', torchvision.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-GPU')"
echo "=== TORCH INSTALL COMPLETE ==="
