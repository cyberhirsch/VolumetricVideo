#!/usr/bin/env bash
# Start the TGH WebSocket render server.
# Then open tgh_viewer.html in a browser (works directly via file://).
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

"$ENV/bin/python" "/mnt/g/3D Generation/Long Volumetric Video/tgh_serve.py"
