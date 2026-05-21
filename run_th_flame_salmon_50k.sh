#!/usr/bin/env bash
# Overnight: 50k-Iter Lauf auf flame_salmon (300 Frames). Selbe Daten/Memory-
# Konfiguration wie der erfolgreiche 3k-Lauf.
#
# Checkpoint-Schema: 10 Epochs x 5000 Iter = 50000 total. Nach jeder Epoche
# wird latest.pt geschrieben (Rolling-Checkpoint), nummerierte Checkpoints
# alle 2 Epochs. Bei einem Crash gehen also maximal ~80 Min verloren statt
# der gesamten Nacht. Eval nur am Ende (eval_ep=10) um Overhead niedrig zu
# halten.
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

cd "$EVC"
echo "=== flame_salmon 50k-iter Lauf (10 epochs x 5000 iter, mit checkpoints) ==="
"$ENV/bin/python" -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussianth/gaussianth_flame_salmon.yaml \
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
    val_dataloader_cfg.dataset_cfg.dataloading_workers=4 \
    2>&1
echo "=== 50K RUN EXIT: $? ==="
