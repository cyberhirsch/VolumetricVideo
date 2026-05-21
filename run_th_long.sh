#!/usr/bin/env bash
# Long TGH training run (single epoch, ~2000 iters) on the example dataset.
# Uses the paper's default densification hyperparameters; random init means the
# absolute PSNR will be below the paper's numbers (which use SfM init), but
# this exercises the full pipeline: densification cycles, SH-degree progression,
# Compact Appearance lambda_h cutoff, hierarchy re-assignment, eval, visualization.
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

cd "$EVC"
echo "=== GaussianTHSampler long run (2000 iters, paper defaults) ==="
"$ENV/bin/python" -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussianth/gaussianth_actor1_4_subseq.yaml \
    runner_cfg.epochs=1 \
    runner_cfg.ep_iter=2000 \
    runner_cfg.eval_ep=1 \
    runner_cfg.save_ep=1 \
    runner_cfg.save_latest_ep=1 \
    runner_cfg.log_interval=100 \
    model_cfg.sampler_cfg.n_init_points=65536 \
    dataloader_cfg.dataset_cfg.ratio=0.5 \
    val_dataloader_cfg.dataset_cfg.ratio=0.5 \
    2>&1
echo "=== LONG RUN EXIT: $? ==="
