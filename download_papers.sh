#!/usr/bin/env bash
# Download related-work papers (arXiv PDFs) into docs/papers/
set -o pipefail

DEST="/mnt/g/3D Generation/Long Volumetric Video/docs/papers"
mkdir -p "$DEST"
cd "$DEST"

# label  arxiv_id
PAPERS=(
    "TGH_Xu2024__2412.09608"
    "Yang2023_4DGS__2310.10642"
    "Wu2024_4DGaussians__2310.08528"
    "MEGA_Zhang2024__2410.13613"
    "ReConGS_2025__2509.24325"
    "InstantGaussianStream_2025__2503.16979"
    "4DGCPro_2025__2509.17513"
    "4DMoDe_2025__2509.17506"
    "VDEGaussian_2025__2508.02129"
    "PackUV_2026__2602.23040"
)

for entry in "${PAPERS[@]}"; do
    label="${entry%__*}"
    arxiv="${entry##*__}"
    out="${label}_arXiv${arxiv}.pdf"
    url="https://arxiv.org/pdf/${arxiv}"
    echo "=== $label ($arxiv) -> $out ==="
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "  already present: $(ls -lh "$out" | awk '{print $5}')"
        continue
    fi
    wget --continue --tries=20 --waitretry=5 --timeout=60 \
         --user-agent="Mozilla/5.0" --no-verbose \
         -O "$out" "$url"
    echo "  got: $(ls -lh "$out" | awk '{print $5}')"
done

echo
echo "=== final ==="
ls -lh "$DEST"
echo "=== PAPERS DOWNLOAD DONE ==="
