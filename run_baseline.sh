#!/usr/bin/env bash
# Phase 1: smoke-test 3DGS+T training end-to-end (short run)
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

cd "$EVC"
echo "=== train smoke test (1 epoch x 50 iters, ratio 0.5) ==="
"$ENV/bin/python" -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussiant/gaussiant_actor1_4_subseq.yaml \
    runner_cfg.epochs=1 \
    runner_cfg.ep_iter=50 \
    runner_cfg.eval_ep=1 \
    runner_cfg.save_ep=1 \
    runner_cfg.save_latest_ep=1 \
    runner_cfg.log_interval=1 \
    model_cfg.sampler_cfg.skip_loading_points=True \
    dataloader_cfg.dataset_cfg.ratio=0.5 \
    val_dataloader_cfg.dataset_cfg.ratio=0.5 \
    2>&1
echo "=== BASELINE SMOKE TEST EXIT: $? ==="
