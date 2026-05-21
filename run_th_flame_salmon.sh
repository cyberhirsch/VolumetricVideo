#!/usr/bin/env bash
# Long run on Neural3DV flame_salmon (the paper's canonical Table 1 benchmark).
# 3000 iters, paper-default adaptive control, 1200 frames at ratio 0.5.
#
# This is the largest training run of the reproduction so far. The TGH's
# per-iter cost is governed by the active segment subset (constant in video
# length), so even 1200 frames stays affordable per iteration. Densification
# runs from iter 500 to 2500 every 100 iters (= 20 events).
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

cd "$EVC"
echo "=== flame_salmon TGH run (3000 iters, 300 frames, ratio 0.5, fewer workers) ==="
"$ENV/bin/python" -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussianth/gaussianth_flame_salmon.yaml \
    runner_cfg.epochs=1 \
    runner_cfg.ep_iter=3000 \
    runner_cfg.eval_ep=1 \
    runner_cfg.save_ep=1 \
    runner_cfg.save_latest_ep=1 \
    runner_cfg.log_interval=100 \
    model_cfg.sampler_cfg.n_init_points=131072 \
    model_cfg.sampler_cfg.densify_until_iter=2500 \
    dataloader_cfg.dataset_cfg.frame_sample=[0,300,1] \
    dataloader_cfg.dataset_cfg.dataloading_workers=8 \
    val_dataloader_cfg.dataset_cfg.frame_sample=[0,300,50] \
    val_dataloader_cfg.dataset_cfg.dataloading_workers=4 \
    2>&1
echo "=== FLAME_SALMON RUN EXIT: $? ==="
