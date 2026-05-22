#!/usr/bin/env bash
# Fresh 50k benchmark for the active-subset TGH conditioning path.
# Matches the prior 300-frame flame_salmon benchmark settings:
# 10 epochs x 5000 iterations, densification from 500 to 15000.
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTHONUNBUFFERED=1

cd "$EVC"
echo "=== active-subset flame_salmon 50k benchmark start: $(date --iso-8601=seconds) ==="
"$ENV/bin/python" -u -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussianth/gaussianth_flame_salmon.yaml \
    exp_name=gaussianth_flame_salmon_active_subset_50k \
    runner_cfg.resume=False \
    runner_cfg.epochs=10 \
    runner_cfg.ep_iter=5000 \
    runner_cfg.save_ep=2 \
    runner_cfg.save_latest_ep=1 \
    runner_cfg.eval_ep=10 \
    runner_cfg.log_interval=500 \
    runner_cfg.save_lim=5 \
    model_cfg.sampler_cfg.n_init_points=131072 \
    model_cfg.sampler_cfg.densify_from_iter=500 \
    model_cfg.sampler_cfg.densify_until_iter=15000 \
    model_cfg.sampler_cfg.densification_interval=100 \
    model_cfg.sampler_cfg.densify_grad_threshold=2.0e-4 \
    model_cfg.sampler_cfg.sh_update_iter=1000 \
    dataloader_cfg.dataset_cfg.frame_sample=[0,300,1] \
    dataloader_cfg.dataset_cfg.dataloading_workers=8 \
    val_dataloader_cfg.dataset_cfg.frame_sample=[0,300,50] \
    val_dataloader_cfg.dataset_cfg.dataloading_workers=4
status=$?
echo "=== active-subset flame_salmon 50k benchmark end: $(date --iso-8601=seconds) ==="
echo "=== ACTIVE_SUBSET_50K RUN EXIT: ${status} ==="
exit "$status"
