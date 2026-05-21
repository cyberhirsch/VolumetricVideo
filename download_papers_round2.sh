#!/usr/bin/env bash
# Zweite Runde: weitere relevante Paper (Generative-/Diffusion-Gaussian-Splatting
# Familie + Original-3DGS), heruntergeladen in docs/papers/.
set -o pipefail

DEST="/mnt/g/3D Generation/Long Volumetric Video/docs/papers"
mkdir -p "$DEST"
cd "$DEST"

# entries: label__url
PAPERS=(
    "3DGS_Kerbl2023_arXiv2308.04079__https://arxiv.org/pdf/2308.04079"
    "DiffSplat_2025_arXiv2501.16764__https://arxiv.org/pdf/2501.16764"
    "GaussianDreamer_Yi2024_arXiv2310.08529__https://arxiv.org/pdf/2310.08529"
    "GGS_Schwarz2025_ICCV_arXiv2503.13272__https://arxiv.org/pdf/2503.13272"
    "TeacherGuidedDiffusion_Peng2025_ICCV__https://openaccess.thecvf.com/content/ICCV2025/papers/Peng_A_Lesson_in_Splats_Teacher-Guided_Diffusion_for_3D_Gaussian_Splats_ICCV_2025_paper.pdf"
    "CompleteGaussianSplats_2025_arXiv2508.21542__https://arxiv.org/pdf/2508.21542"
    "HuGDiffusion_2025_arXiv2501.15008__https://arxiv.org/pdf/2501.15008"
    "GaussianMotion_2025_arXiv2502.11642__https://arxiv.org/pdf/2502.11642"
    "3D-LATTE_OpenReview_RBuVfiOFvI__https://openreview.net/pdf?id=RBuVfiOFvI"
    "BakingGSIntoDiffusionDenoiser_Cai2025_ICCV__https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_Baking_Gaussian_Splatting_into_Diffusion_Denoiser_for_Fast_and_Scalable_ICCV_2025_paper.pdf"
)

for entry in "${PAPERS[@]}"; do
    label="${entry%__*}"
    url="${entry##*__}"
    out="${label}.pdf"
    echo "=== $label ==="
    if [ -f "$out" ] && [ -s "$out" ]; then
        echo "  already present: $(ls -lh "$out" | awk '{print $5}')"
        continue
    fi
    wget --continue --tries=20 --waitretry=5 --timeout=60 \
         --user-agent="Mozilla/5.0" --no-verbose \
         -O "$out" "$url"
    if [ ! -s "$out" ]; then
        echo "  ! empty/failed, removing"
        rm -f "$out"
    else
        echo "  got: $(ls -lh "$out" | awk '{print $5}')"
    fi
done

echo
echo "=== full papers/ listing ==="
ls -lh "$DEST"
echo "=== PAPERS DOWNLOAD ROUND 2 DONE ==="
