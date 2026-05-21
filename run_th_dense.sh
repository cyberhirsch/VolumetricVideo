#!/usr/bin/env bash
# Phase 6: train TGH long enough to trigger densification (300 iters,
# densify from iter 50 every 50 iters -> 5 densify events).
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

cd "$EVC"
echo "=== GaussianTHSampler densification test (1 ep x 300 iters) ==="
"$ENV/bin/python" -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussianth/gaussianth_actor1_4_subseq.yaml \
    runner_cfg.epochs=1 \
    runner_cfg.ep_iter=300 \
    runner_cfg.eval_ep=1 \
    runner_cfg.save_ep=1 \
    runner_cfg.save_latest_ep=1 \
    runner_cfg.log_interval=20 \
    model_cfg.sampler_cfg.densify_from_iter=50 \
    model_cfg.sampler_cfg.densification_interval=50 \
    model_cfg.sampler_cfg.densify_grad_threshold=1.0e-5 \
    model_cfg.sampler_cfg.n_init_points=32768 \
    dataloader_cfg.dataset_cfg.ratio=0.5 \
    val_dataloader_cfg.dataset_cfg.ratio=0.5 \
    2>&1
echo "=== DENSIFICATION TEST EXIT: $? ==="
