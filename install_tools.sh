#!/usr/bin/env bash
# Retry-able install of zip / unzip / ffmpeg (and p7zip for safety) into the lvv env.
set -o pipefail

CONDA="$HOME/miniconda3/bin/conda"
ENV="$HOME/miniconda3/envs/lvv"

echo "=== installing zip/unzip/ffmpeg/p7zip ==="
for attempt in 1 2 3 4 5 6 7 8; do
    echo "--- conda attempt $attempt ---"
    if "$CONDA" install -y -n lvv -c conda-forge zip unzip ffmpeg p7zip; then
        echo "tools installed"; break
    fi
    echo "attempt $attempt failed, retrying..."; sleep 5
done

echo "=== verifying ==="
"$ENV/bin/zip" --version 2>&1 | head -1
"$ENV/bin/unzip" -v 2>&1 | head -1
"$ENV/bin/ffmpeg" -version 2>&1 | head -1
"$ENV/bin/7z" --help 2>&1 | head -2
echo "=== TOOLS DONE ==="
