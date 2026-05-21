#!/usr/bin/env bash
# Overnight: 50k-Iter Lauf auf flame_salmon mit der VOLLEN 1200-Frame-Sequenz
# (= dem kanonischen Paper-Setup für Tabelle 1).
#
# Memory-Budget:
#   - 19 cams × 1200 frames × ~750 KB JPEG ≈ 17 GB in RAM für die Dataset-Bytes
#   - mit 4 workers Fork-Overhead ≈ ~25-30 GB peak
#   - WSL hat 56 GB allokiert → sollte passen, aber dichter am Limit als der
#     300-Frame-Lauf (war 17 GB peak von 56)
#   - VRAM: 4× temporale Komplexität kann Gaussian-Count auf 5-10M treiben,
#     das ist bei ~600 bytes/Gaussian ≈ 3-6 GB plus Adam-State 2x ≈ 12-18 GB
#     VRAM; auf 24 GB tight aber machbar
#
# Checkpoint-Schema: 10 Epochs x 5000 Iter (wie der 300-Frame-Lauf), latest.pt
# jede Epoche, nummerierte Snapshots alle 2 Epochs. Bei Crash maximal eine
# Epoche (~1-2h) verloren.
set -o pipefail

ENV="$HOME/miniconda3/envs/lvv"
EVC="/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

cd "$EVC"
echo "=== flame_salmon 50k-iter Lauf (FULL 1200 frames, ratio 0.5) ==="
"$ENV/bin/python" -m easyvolcap.scripts.main -t train \
    -c configs/exps/gaussianth/gaussianth_flame_salmon.yaml \
    exp_name=gaussianth_flame_salmon_1200 \
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
    dataloader_cfg.dataset_cfg.frame_sample=[0,1200,1] \
    dataloader_cfg.dataset_cfg.dataloading_workers=4 \
    val_dataloader_cfg.dataset_cfg.frame_sample=[0,1200,200] \
    val_dataloader_cfg.dataset_cfg.dataloading_workers=2 \
    2>&1
echo "=== 50K 1200-FRAME RUN EXIT: $? ==="
