# Installation Protocol

Exact record of *what was installed where* to make the Temporal Gaussian
Hierarchy reproduction runnable on this machine. Companion to
[`reproduction-methodology.md`](reproduction-methodology.md).

The objective was a fully **userspace** install (no `sudo`), so the host
Windows system was not touched.

---

## 0. Host system

| Item | Value |
|---|---|
| OS | Windows 11 (host) |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| NVIDIA driver | 596.36 (supports CUDA 12.x via WSL passthrough) |
| Visual Studio | 2019 + 2022 (build tools present but not used; the WSL toolchain is used instead) |
| WSL | WSL2, default distro Ubuntu-22.04 (pre-installed) |
| Project root (Windows) | `G:\3D Generation\Long Volumetric Video\` (note: contains spaces — see workaround in §7) |
| Project root (WSL) | `/mnt/g/3D Generation/Long Volumetric Video/` |

WSL probe:

```
$ uname -a
Linux Threadripper 6.6.114.1-microsoft-standard-WSL2 ...
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
NVIDIA GeForce RTX 3090, 24576 MiB, 596.36
$ python3 --version
Python 3.10.12         # system python, NOT used; we use a conda env
```

---

## 1. Source repositories (cloned on Windows)

Both into `G:\3D Generation\Long Volumetric Video\`:

```
git clone --recursive https://github.com/zju3dv/EasyVolcap.git
git clone           https://github.com/dendenxu/fast-gaussian-rasterization.git
```

Resulting tree:

```
G:\3D Generation\Long Volumetric Video\
├── EasyVolcap\                           # MIT license
├── fast-gaussian-rasterization\          # MIT license
├── configs, code, scripts...             # ours
└── docs\                                 # ours
```

Both are visible from WSL as `/mnt/g/3D Generation/Long Volumetric Video/...`
which is editable from Windows tools while runnable from WSL.

---

## 2. Python environment (WSL, fully userspace)

Strategy: **Miniconda installed into `~/miniconda3`, conda env `lvv` with
Python 3.10**. No `sudo` was used anywhere.

### 2.1 Miniconda

Downloaded and installed inside WSL:

```
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda --version
# conda 26.3.2
```

### 2.2 Conda channel Terms of Service (conda ≥ 23)

Newer conda refuses to install from the default channels until ToS is
accepted:

```
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

### 2.3 Environment

```
~/miniconda3/bin/conda create -y -n lvv python=3.10
```

Resulting env root: `~/miniconda3/envs/lvv/`.

### 2.4 CUDA toolkit + GNU toolchain (into the env, not the system)

The conda nvidia channel publishes the full CUDA toolkit so `nvcc` lives in
the env, never in `/usr/local/cuda`:

```
~/miniconda3/bin/conda config --set remote_max_retries 10
~/miniconda3/bin/conda config --set remote_backoff_factor 2
~/miniconda3/bin/conda config --set remote_connect_timeout_secs 60
~/miniconda3/bin/conda config --set remote_read_timeout_secs 120

~/miniconda3/bin/conda install -y -n lvv -c nvidia/label/cuda-12.1.1 cuda-toolkit
~/miniconda3/bin/conda install -y -n lvv -c conda-forge gxx=12 gcc=12 ninja
```

Result:

```
~/miniconda3/envs/lvv/bin/nvcc --version
# Cuda compilation tools, release 12.1, V12.1.105
```

The conda env shadows the host. From here on **every command uses
`$HOME/miniconda3/envs/lvv/bin/...`** explicitly (the WSL ↔ Windows shell
boundary mangles `PATH`-based activation; see §7).

### 2.5 CUDA-enabled PyTorch

The download of the ~760 MB `torch` wheel from `download.pytorch.org` was
the most fragile step on this network — pip's HTTPS transport gave repeated
`[SSL] record layer failure` mid-download. Worked around with
`wget --continue` (resumable across drops), then `pip install --no-deps` of
the local files:

```
ENV=$HOME/miniconda3/envs/lvv
DL=$HOME/wheels && mkdir -p "$DL"
BASE=https://download.pytorch.org/whl/cu121

wget --continue --tries=50 --waitretry=5 --timeout=60 --read-timeout=60 \
     -O "$DL/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl"        "$BASE/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget --continue --tries=50 -O "$DL/torchvision-0.19.1+cu121-cp310-cp310-linux_x86_64.whl" "$BASE/torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl"
wget --continue --tries=50 -O "$DL/torchaudio-2.4.1+cu121-cp310-cp310-linux_x86_64.whl"  "$BASE/torchaudio-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl"

$ENV/bin/pip install --no-deps "$DL"/*.whl
```

### 2.6 PyTorch CUDA runtime wheels (from PyPI)

`--no-deps` skipped the `nvidia-*-cu12` runtime libs (cuBLAS, cuDNN 9, …)
needed by the torch wheel. Installed from PyPI with pip retries (PyPI was
not affected by the same SSL fault as `download.pytorch.org`):

```
$ENV/bin/pip install --retries 10 --timeout 120 \
    nvidia-cublas-cu12==12.1.3.1       nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105   nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==9.1.0.70        nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106     nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106   nvidia-nccl-cu12==2.20.5 \
    nvidia-nvtx-cu12==12.1.105         triton==3.0.0
```

Verification:

```
$ENV/bin/python -c "
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1000,1000, device='cuda'); print((x @ x).sum().item() != 0)
"
# 2.4.1+cu121 True NVIDIA GeForce RTX 3090
# True
```

---

## 3. Compiled CUDA / OpenGL extensions

All built **with the conda toolchain** (the env's `nvcc`, conda's `gcc`/`g++`
12), targeted at RTX 3090 = sm_86. Two env vars matter at build time:

```
export CUDA_HOME=$HOME/miniconda3/envs/lvv
export TORCH_CUDA_ARCH_LIST="8.6"
```

All extensions are installed with `pip install --no-build-isolation` so the
build sees the env's `torch` (build isolation would otherwise spin up a clean
env without torch, breaking the extension's `setup.py`).

| Extension | Origin | Build cost | Result |
|---|---|---|---|
| `simple_knn` | `git+https://gitlab.inria.fr/bkerbl/simple-knn.git` | ~1 min CUDA | KNN distance used in Gaussian scale init |
| `diff_gauss` | `git+https://github.com/dendenxu/diff-gaussian-rasterization` | ~3 min CUDA | Differentiable 3-D Gaussian rasterizer used at train time |
| `fast_gauss` | PyPI: `fast_gauss` | seconds (shader-only) | Hardware-accelerated rasterizer for inference (not yet wired up) |
| `tinycudann` | `git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch` | ~12 min CUDA | Needed by some `PointPlanesSampler` defaults (KPlanes embedders) |

### 3.1 The `-lcuda` linker fix (tinycudann)

The first attempt at `tinycudann` failed at link time:

```
/usr/bin/ld: cannot find -lcuda: No such file or directory
```

`libcuda.so` is the **driver** library, not part of the CUDA toolkit. In WSL
it lives at `/usr/lib/wsl/lib/libcuda.so` (provided by the Windows NVIDIA
driver); the conda env additionally ships a stub at
`$ENV/lib/stubs/libcuda.so`. Either is enough for *linking*. Fix by
exporting `LIBRARY_PATH` before `pip install`:

```
export LIBRARY_PATH="$ENV/lib/stubs:/usr/lib/wsl/lib:$LIBRARY_PATH"
```

After that, all four extensions build cleanly.

Final import sanity:

```
$ENV/bin/python -c "
import torch, simple_knn._C, diff_gauss, fast_gauss, tinycudann
print('all extensions OK with torch loaded')
"
```

---

## 4. EasyVolcap install + Python deps

`pip install -e .` of the repo on `/mnt/g`, with isolation disabled so it
uses the env's already-built setuptools / torch.

```
# core deps from EasyVolcap requirements.txt (per-package, with retry; one
# entry — imgui-bundle — fails to build a wheel on this box, harmless since
# only the GUI viewer needs it)
$ENV/bin/pip install av tqdm pdbr h5py yapf ujson PyGLM scipy sympy addict \
                    PyYaml psutil ipython trimesh imageio PyOpenGL pycolmap \
                    PyMCubes pyperclip pyntcloud websockets PyTurboJPEG    \
                    tensorboard ruamel.yaml cuda-python scikit-image       \
                    opencv-python plyfile lpips

# libjpeg-turbo from conda-forge (system lib needed by PyTurboJPEG)
~/miniconda3/bin/conda install -y -n lvv -c conda-forge libjpeg-turbo

# editable install of EasyVolcap itself
$ENV/bin/pip install --no-build-isolation --no-deps -e \
    "/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
```

### 4.1 `pkg_resources` regression (setuptools ≥ 81)

EasyVolcap's data / sampler modules import `pkg_resources`. Setuptools 81
removed this module from the default install. Without it, the imports cascade
into `KeyError: VolumetricVideoDataloader is not in the dataloaders registry`
at runtime. Fix:

```
$ENV/bin/pip install "setuptools<81"
```

Verification:

```
$ENV/bin/python -c "import pkg_resources; print('pkg_resources OK')"
```

### 4.2 Other small extras

`glfw` is installed even though the GUI viewer isn't used — it appears as a
fail-fast import early in the runner. `gdown` is installed for the example
dataset download (§5).

```
$ENV/bin/pip install glfw gdown
```

---

## 5. Dataset

The EasyVolcap example dataset (ENeRF-Outdoor subset, file id
`1XxeO7TnAPvDugnxguEF5Jp89ERS9CAia`, ~267 MB) was downloaded to the **WSL
native filesystem** for fast training I/O (the `/mnt/g` 9P mount has
substantial per-file overhead):

```
mkdir -p $HOME/lvv-data && cd $HOME/lvv-data
$ENV/bin/gdown "https://drive.google.com/uc?id=1XxeO7TnAPvDugnxguEF5Jp89ERS9CAia" -O example_dataset.zip
$ENV/bin/python -c "import zipfile; zipfile.ZipFile('example_dataset.zip').extractall('.')"
mkdir -p enerf_outdoor && mv actor1_4_subseq enerf_outdoor/
```

Resulting layout (18 views × 30 frames):

```
$HOME/lvv-data/
└── enerf_outdoor/
    └── actor1_4_subseq/
        ├── images/
        │   ├── 00/000000.jpg ... 000029.jpg
        │   ├── 01/...
        │   └── 17/...
        ├── intri.yml
        └── extri.yml
```

Symlinked into the EasyVolcap repo so config-relative paths resolve:

```
ln -s $HOME/lvv-data "/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap/data"
```

Trained models and eval outputs land under `$HOME/lvv-data/trained_model/`
and `$HOME/lvv-data/result/` respectively (same symlink).

---

## 6. Patches applied to EasyVolcap

One patch only, for `pycolmap` API drift. See
[`reproduction-methodology.md` §6](reproduction-methodology.md) for the math.

Modified file: `EasyVolcap/easyvolcap/utils/undist_utils.py`
- Added a private helper `_img_from_cam(cam, pts)` that pads 2-D
  normalized-image-plane points to 3-D (z = 1) before calling
  `Camera.img_from_cam` (the newer pycolmap signature requires the 3-D
  overload).
- Replaced the four `undistorted_camera.img_from_cam(...)` call sites with
  `_img_from_cam(undistorted_camera, ...)`.

---

## 7. Workarounds for the WSL ⇄ Windows boundary

Two environment quirks repeatedly bit shell scripting; each was worked
around rather than configured away.

### 7.1 Project path contains spaces

`G:\3D Generation\Long Volumetric Video\` becomes
`/mnt/g/3D Generation/Long Volumetric Video/` in WSL. EasyVolcap's
`evc-train` console-script launcher builds an unquoted shell command from
that path, which then word-splits. Symptom:

```
$ evc-train -c configs/...yaml
python: can't open file '/mnt/g/3D': [Errno 2] No such file or directory
```

Workaround: bypass the launcher and invoke the module directly so no file
path is on the shell line:

```
$ENV/bin/python -m easyvolcap.scripts.main -t train -c <config> <overrides...>
```

All run scripts in the project root use this form.

### 7.2 `wsl.exe ... bash -c '...'` mangles intermediate variables

When the Git-Bash shell on Windows invokes
`wsl.exe -d Ubuntu-22.04 -- bash -c 'CONDA=$HOME/...; $CONDA ...'`,
intermediate variable assignments inside the single-quoted string are
sometimes empty by the time they are used. `$HOME` is always populated by
the WSL shell, but custom assignments are unreliable. Workaround: never set
intermediate variables in one-liners; write a `setup_*.sh` script in the
project root and execute it as `wsl.exe -d Ubuntu-22.04 -- bash
"/mnt/g/.../script.sh"`. All multi-step install steps use this pattern.

---

## 8. Install scripts (project root, idempotent)

Each script can be re-run safely. They live next to the repos on the host
filesystem and are invoked from WSL via the absolute `/mnt/g/...` path.

| Script | What it does |
|---|---|
| `setup_env.sh` | Miniconda → conda env `lvv` (Python 3.10) → ToS accept |
| `setup_env2.sh` | Retry-able CUDA toolkit + gcc 12 + ninja + PyTorch install |
| `setup_torch.sh` | Resumable `wget` of `torch` / `torchvision` / `torchaudio` wheels, then local `pip install --no-deps` |
| `setup_torchdeps.sh` | `nvidia-*-cu12` runtime wheels + Triton from PyPI, with retries |
| `setup_extensions.sh` | Build `simple_knn`, `diff_gauss`, `fast_gauss` with `--no-build-isolation` |
| `setup_tcnn.sh` | Build `tinycudann` (with `LIBRARY_PATH` fix for `-lcuda`) |
| `setup_easyvolcap.sh` | EasyVolcap core deps + editable install |
| `setup_dataset.sh` | `gdown` the example dataset, unpack, symlink |
| `run_baseline.sh` | Smoke-test the stock 3DGS+T sampler (validation of the framework) |
| `run_th.sh` | Smoke-test our `GaussianTHSampler` (50 iters) |
| `run_th_dense.sh` | Densification smoke test (300 iters, paper-default adaptive control) |
| `run_th_long.sh` | The 2000-iter long run that produced the documented results |

---

## 9. Where everything lives — quick reference

| Item | Path |
|---|---|
| WSL home | `/home/cyberhirsch/` |
| Conda root | `~/miniconda3/` |
| Python env | `~/miniconda3/envs/lvv/` |
| Python | `~/miniconda3/envs/lvv/bin/python` (Python 3.10) |
| nvcc | `~/miniconda3/envs/lvv/bin/nvcc` (CUDA 12.1) |
| g++ | `~/miniconda3/envs/lvv/bin/g++` (12.x) |
| PyTorch wheels cache | `~/wheels/` |
| Dataset | `~/lvv-data/enerf_outdoor/actor1_4_subseq/` |
| Trained models | `~/lvv-data/trained_model/` |
| Eval outputs | `~/lvv-data/result/` |
| EasyVolcap repo (editable install source) | `/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap/` |
| `data` symlink inside EasyVolcap | `EasyVolcap/data → ~/lvv-data/` |
| `libcuda.so` (driver, link-time) | `/usr/lib/wsl/lib/libcuda.so` (WSL passthrough) |
| `libcuda.so` (conda stub, link-time fallback) | `~/miniconda3/envs/lvv/lib/stubs/libcuda.so` |

---

## 10. Reproducing the environment from scratch

```
# 0. clone repos on Windows
cd "G:\3D Generation\Long Volumetric Video"
git clone --recursive https://github.com/zju3dv/EasyVolcap.git
git clone           https://github.com/dendenxu/fast-gaussian-rasterization.git

# 1. all the rest from inside WSL
wsl -d Ubuntu-22.04
PROJ='/mnt/g/3D Generation/Long Volumetric Video'

bash "$PROJ/setup_env.sh"            # miniconda + env
bash "$PROJ/setup_env2.sh"           # CUDA toolkit + gcc + (attempted) torch
bash "$PROJ/setup_torch.sh"          # resumable torch wheels
bash "$PROJ/setup_torchdeps.sh"      # nvidia-*-cu12 runtime libs
bash "$PROJ/setup_extensions.sh"     # diff_gauss, simple_knn, fast_gauss
bash "$PROJ/setup_tcnn.sh"           # tinycudann (with -lcuda fix)
bash "$PROJ/setup_easyvolcap.sh"     # EasyVolcap deps + editable install
bash "$PROJ/setup_dataset.sh"        # example dataset + symlink

# 2. smoke test (50 iters)
bash "$PROJ/run_th.sh"

# 3. long run (2000 iters, the documented results)
bash "$PROJ/run_th_long.sh"
```

Total fresh-install wall-clock from a working Ubuntu-22.04 WSL distro:
roughly 35–50 minutes (~15 min of CUDA extension compilation, ~15 min of
network downloads for CUDA toolkit + torch + nvidia runtime wheels +
dataset, plus shell time). On a slower network expect the resumable wget
retries to dominate.
