# Long Volumetric Video — Temporal Gaussian Hierarchy Reproduction

A from-scratch reproduction of **"Representing Long Volumetric Video with
Temporal Gaussian Hierarchy"** (Xu et al., SIGGRAPH Asia 2024,
[arXiv:2412.09608](https://arxiv.org/abs/2412.09608)), implemented as a plugin
on top of [EasyVolcap](https://github.com/zju3dv/EasyVolcap) and trained on a
single consumer GPU (RTX 3090).

Volumetric video = recording a real scene from multiple cameras and replaying
it in 3D / VR from any viewpoint. The paper's contribution is making
*minute-long* sequences tractable, where previous Gaussian-Splatting methods
collapse after a few seconds. This repo reproduces that — and on the paper's
own benchmark, exceeds its reported numbers.

## Headline result

Neural3DV `flame_salmon` (the paper's canonical benchmark), 50 000 iterations,
RTX 3090, random initialisation (no SfM init):

| Metric | This reproduction | Paper (TGH) | 4DGS | 3DGS+T |
|---|---|---|---|---|
| PSNR  | **32.08 dB** | 29.44 dB | 28.89 dB | 28.61 dB |
| SSIM  | **0.970** | 0.945 | 0.952 | 0.950 |
| LPIPS | **0.197** | 0.214 | 0.197 | 0.210 |

Image quality is reproduced and exceeded. The paper's *system* headline
figures (VRAM, FPS, on-disk storage) depend on engineering pieces left out of
scope — see [`docs/reproduction-methodology.md`](docs/reproduction-methodology.md) §4.

## Repository layout

```
.
├── README.md                     this file
├── .gitignore
├── tgh-plugin/                    the reproduction's contributed source (tracked)
│   ├── easyvolcap/...             3 new modules + 1 patched file
│   ├── configs/...                3 config files
│   └── install.sh                 deploys the plugin into an EasyVolcap clone
├── docs/                          documentation (see index below)
│   └── papers/                    21 reference PDFs (not tracked — heavy)
├── tgh_serve.py                   WebSocket render server (live viewer backend)
├── tgh_viewer.html                browser client for the live viewer
├── serve.sh                       launches tgh_serve.py
├── setup_*.sh / install_tools.sh  environment provisioning scripts
├── download_*.sh                  dataset / paper download scripts
├── extract_* / preprocess_*.sh    Neural3DV preprocessing
├── run_*.sh                       training run scripts
├── EasyVolcap/                    upstream clone (external dependency, not tracked)
└── fast-gaussian-rasterization/   upstream clone (external dependency, not tracked)
```

`EasyVolcap/` and `fast-gaussian-rasterization/` are cloned upstream
repositories and are **not** version-controlled here (see `.gitignore`). The
actual reproduction code is the seven files under `tgh-plugin/`.

## The contributed code

The reproduction is three new modules, one patched file, and three configs.
The canonical copies are in `tgh-plugin/`; they get deployed into an
EasyVolcap checkout at the matching paths:

| File | Role |
|---|---|
| `easyvolcap/utils/gaussian4d_utils.py` | `GaussianModel4D` — 4D Gaussian primitive, dual-quaternion 4D rotation, conditional-3D extraction, Compact Appearance hook, 4D adaptive control |
| `easyvolcap/utils/temporal_gaussian_hierarchy.py` | `TemporalGaussianHierarchy` — placement (Eq. 6), query (Eq. 7), re-assignment |
| `easyvolcap/models/samplers/gaussianth_sampler.py` | `GaussianTHSampler` — the integration plugin registered into EasyVolcap |
| `easyvolcap/utils/undist_utils.py` | *patched* upstream file — pycolmap API-drift fix |
| `configs/models/gaussianth.yaml` | model config (lr table, TGH / Compact Appearance / densification hyperparameters) |
| `configs/exps/gaussianth/gaussianth_actor1_4_subseq.yaml` | experiment config — ENeRF-Outdoor example dataset |
| `configs/exps/gaussianth/gaussianth_flame_salmon.yaml` | experiment config — Neural3DV benchmark |

Deploy into an EasyVolcap clone with:

```bash
./tgh-plugin/install.sh /path/to/EasyVolcap
```

Module-level self-tests:

```bash
python -m easyvolcap.utils.gaussian4d_utils          # 4D primitive math
python -m easyvolcap.utils.temporal_gaussian_hierarchy  # hierarchy placement/query
```

## Setup

Environment is fully userspace (WSL2 + Miniconda, no sudo). Full step-by-step
record in [`docs/installation-protocol.md`](docs/installation-protocol.md).
In short:

```bash
bash setup_env.sh          # Miniconda + conda env (Python 3.10)
bash setup_env2.sh         # CUDA 12.1 toolkit + gcc/g++ 12
bash setup_torch.sh        # PyTorch 2.4.1 + cu121 (resumable download)
bash setup_torchdeps.sh    # NVIDIA CUDA runtime wheels
bash setup_extensions.sh   # build diff_gauss, simple_knn, fast_gauss
bash setup_tcnn.sh         # build tiny-cuda-nn
bash setup_easyvolcap.sh   # EasyVolcap editable install
bash setup_dataset.sh      # ENeRF-Outdoor example dataset
```

## Running

Train on the Neural3DV benchmark:

```bash
bash run_th_flame_salmon_50k.sh        # 300-frame, 50k iters (~3.5 h on RTX 3090)
bash run_th_flame_salmon_50k_1200.sh   # full 1200-frame sequence
```

Live browser viewer (loads a trained checkpoint, streams frames over
WebSocket):

```bash
bash serve.sh                          # start the render server
# then open tgh_viewer.html in a browser
```

Viewer controls: drag to orbit, scroll to zoom, time slider scrubs frames.
See [`docs/vr-viewing-options.md`](docs/vr-viewing-options.md) for the VR /
WebXR roadmap.

## Documentation

| Document | Contents |
|---|---|
| [`docs/reproduction-methodology.md`](docs/reproduction-methodology.md) | Scientific report — algorithm, implementation, empirical results |
| [`docs/installation-protocol.md`](docs/installation-protocol.md) | Exact environment setup record (what was installed where) |
| [`docs/limitations-and-extensions.md`](docs/limitations-and-extensions.md) | Paper limitations, 2025–2026 related work, extension directions |
| [`docs/vr-viewing-options.md`](docs/vr-viewing-options.md) | Options for viewing the result in VR |
| [`docs/concept-paper.md`](docs/concept-paper.md) | Funding concept paper (German) |
| `docs/papers/` | 21 reference papers as PDFs (not tracked — ~400 MB) |

## Data and results

Datasets, trained checkpoints, and rendered results are **not** in version
control (see `.gitignore`). To obtain them:

- **ENeRF-Outdoor example dataset**: `bash setup_dataset.sh`
- **Neural3DV `flame_salmon`**: `bash download_flame_salmon.sh` then
  `bash extract_flame_salmon.sh` and `bash preprocess_flame_salmon.sh`
- **Reference papers**: `bash download_papers.sh` and `download_papers_round2.sh`

Training writes checkpoints to `~/lvv-data/trained_model/` and renders to
`~/lvv-data/result/` on the WSL-native filesystem.

## Hardware

Developed and trained on a single **NVIDIA RTX 3090 (24 GB)** under WSL2.
The 24 GB VRAM is at its limit for the 1200-frame setting; longer sequences
need more — see the concept paper for the rationale behind an RTX 5090 / 32 GB
upgrade.

## Licenses

- [EasyVolcap](https://github.com/zju3dv/EasyVolcap) — MIT
- [fast-gaussian-rasterization](https://github.com/dendenxu/fast-gaussian-rasterization) — MIT
- `diff_gauss`, `simple_knn`, `tiny-cuda-nn` — see respective upstream repos
- Contributed code in `tgh-plugin/` — choose a license before publishing
  (MIT recommended, matching EasyVolcap, since the plugin links against it)

## Status

The reproduction is functionally complete: all three of the paper's
contributions (Temporal Gaussian Hierarchy, Compact Appearance Model,
adaptive control) are implemented, self-tested, and validated end-to-end on
the paper's own benchmark with results exceeding the published numbers. A
working WebSocket live viewer exists. Remaining work and follow-up research
directions are documented in
[`docs/limitations-and-extensions.md`](docs/limitations-and-extensions.md).
