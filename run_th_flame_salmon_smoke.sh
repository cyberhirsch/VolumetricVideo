#!/usr/bin/env bash
# Smoke test: 200 iters on a small subset (first 60 frames) of flame_salmon
# to confirm the data loaders, hierarchy at 0.25 normalized segment, and
# rendering all work before kicking off the long run.
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

cd "$EVC"
echo "=== flame_salmon TGH smoke (200 iters, 60-frame slice, ratio 0.5) ==="
"$ENV/bin/python" -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussianth/gaussianth_flame_salmon.yaml \
    runner_cfg.epochs=1 \
    runner_cfg.ep_iter=200 \
    runner_cfg.eval_ep=1 \
    runner_cfg.save_ep=1 \
    runner_cfg.save_latest_ep=1 \
    runner_cfg.log_interval=20 \
    dataloader_cfg.dataset_cfg.frame_sample=[0,60,1] \
    val_dataloader_cfg.dataset_cfg.frame_sample=[0,60,30] \
    model_cfg.sampler_cfg.n_init_points=65536 \
    model_cfg.sampler_cfg.densify_from_iter=10000 \
    2>&1
echo "=== SMOKE EXIT: $? ==="
