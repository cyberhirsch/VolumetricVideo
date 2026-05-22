#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/g/3D Generation/Long Volumetric Video"
LOG="$ROOT/results/train_300frames_active_subset.log"
PIDFILE="$ROOT/results/train_300frames_active_subset.pid"

mkdir -p "$ROOT/results"
cd "$ROOT"

nohup bash run_th_flame_salmon.sh > "$LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PIDFILE"
echo "started pid=$pid log=$LOG"
