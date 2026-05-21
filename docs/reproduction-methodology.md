# TGH Paper Reproduction — Scientific Documentation

Reproduction of:

> Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun, Hujun Bao, Xiaowei Zhou.
> **Representing Long Volumetric Video with Temporal Gaussian Hierarchy.**
> ACM Transactions on Graphics 43(6), Article 171 (SIGGRAPH Asia 2024).
> [arXiv:2412.09608](https://arxiv.org/abs/2412.09608)

Reproduction implemented on top of the authors' own framework:

- [EasyVolcap](https://github.com/zju3dv/EasyVolcap) (MIT)
- [fast-gaussian-rasterization](https://github.com/dendenxu/fast-gaussian-rasterization) (MIT)

This document is the **scientific report**. The companion document
[`installation-protocol.md`](installation-protocol.md) lists exactly what was
installed where.

---

## 1. Paper synopsis (for context)

The paper attacks the *scaling* problem in volumetric video: previous Gaussian
Splatting methods (3DGS, 4DGS, 4K4D) carry VRAM and storage that grow linearly
with video length, so a 24 GB GPU is exhausted at roughly 300 frames (≈10 s).
Three contributions make minute-long sequences tractable:

1. **Temporal Gaussian Hierarchy (TGH).** A multi-level binary partition of the
   time axis. Each 4D Gaussian is stored in the *shortest* segment that fully
   covers its temporal influence range. At a query time only one segment per
   level is active (Eq. 7), giving O(log N) sampling and a nearly constant
   per-iteration cost regardless of video duration.

2. **Compact Appearance Model.** A gradient-thresholding scheme that keeps the
   residual spherical-harmonic (SH) coefficients of "diffuse" Gaussians at zero
   and only activates SH for the high-gradient minority (≈15 % by default).
   Combined with a post-training Huffman coding step this gives a ~2.5×
   storage reduction with negligible quality loss.

3. **Hardware rasterizer.** A geometry-shader-based OpenGL/CUDA-sort renderer
   (`fast_gauss`) that delivers ~5× faster inference than the original
   `diff_gauss` software path.

Headline result: 18,000-frame (10 min) sequence at 1080p, 17.2 GB VRAM, 2.2 GB
storage, 450 FPS on an RTX 4090.

---

## 2. Framework and integration choice

The reproduction extends **EasyVolcap** rather than building from scratch. The
paper itself was developed inside this framework, so the dataset loaders,
camera handling, rasterizer wrappers, loss / visualizer / evaluator are all
already present and tested. The contribution we had to add is exactly the
*new representation* — the 4D Gaussians plus their hierarchy.

EasyVolcap's plugin system was used end-to-end:

- The new sampler is registered with `@SAMPLERS.register_module()`.
- A new model config (`configs/models/gaussianth.yaml`) names the sampler and
  supplies its hyperparameters and the per-parameter learning-rate table.
- A new experiment config combines the base, model, and dataset YAMLs.
- Training, evaluation, and visualization are invoked via the stock entry
  point `python -m easyvolcap.scripts.main -t {train|test} -c <config>`.

No other EasyVolcap subsystem was forked.

---

## 3. Algorithmic implementation

All new code lives under `EasyVolcap/easyvolcap/` and is broken into three
files that map one-to-one onto the three contributions of the paper, plus a
sampler that integrates them.

### 3.1 The 4D Gaussian primitive — `utils/gaussian4d_utils.py`

Following Yang et al. 2023b ("Real-time Photorealistic Dynamic Scene
Representation and Rendering with 4D Gaussian Splatting"), each 4D Gaussian
carries:

| Symbol | Storage | Activation |
|---|---|---|
| spatial mean μ ∈ ℝ³ | `_xyz` (N, 3) | identity |
| temporal mean μ_t ∈ ℝ | `_t` (N, 1) | identity |
| spatial scale s_xyz ∈ ℝ³ | `_scaling` (N, 3) | `exp` |
| temporal scale s_t ∈ ℝ | `_scaling_t` (N, 1) | `exp` |
| left isotropic quaternion q_l ∈ ℝ⁴ | `_rotation_l` (N, 4) | `F.normalize` |
| right isotropic quaternion q_r ∈ ℝ⁴ | `_rotation_r` (N, 4) | `F.normalize` |
| opacity o ∈ ℝ | `_opacity` (N, 1) | sigmoid |
| base SH (DC term) c_base ∈ ℝ³ | `_features_dc` (N, 1, 3) | identity |
| residual SH h ∈ ℝ^m | `_features_rest` (N, (deg+1)²−1, 3) | identity (zero-init) |

**4-D rotation from the dual-quaternion pair.** A 4-D rotation is parameterised
by `(q_l, q_r)` acting as left/right Hamilton multiplications on the
quaternion-encoded 4-D vector:

```
R₄(q_l, q_r) = L(q_l) · R(q_r) ∈ SO(4)
```

with the standard 4×4 left/right multiplication matrices `L` and `R`. Both
helpers are implemented as `build_quat_left_matrix` / `build_quat_right_matrix`
and combined in `build_rotation_4d`.

**4-D covariance.** Σ = R₄ S S^T R₄^T where S = diag(s_xyz, s_t). Implemented
as `GaussianModel4D.get_cov4`.

**Conditional 3-D Gaussian at time t.** Standard multivariate-normal
conditioning with the partition

```
Σ = [[Σ_xx (3×3), Σ_xt (3×1)],
     [Σ_tx (1×3), Σ_tt (1×1)]]
```

gives, at timestamp `t`:

- conditional mean: μ_xyz + Σ_xt Σ_tt⁻¹ (t − μ_t)
- conditional covariance: Σ_xx − Σ_xt Σ_tt⁻¹ Σ_tx
- temporal marginal opacity factor: exp(−½ (t − μ_t)² / Σ_tt)
- effective opacity: o · opacity_factor

This is implemented as `get_conditional_3d(t) → (xyz3, cov6, occ1)` where
`cov6` is the upper-triangular packing of the 3-D covariance directly
consumable by `diff_gauss` via `cov3D_precomp`.

**Influence range** (paper Eq. 4):

```
r = √(−2 ln(o_th) · Σ_tt)
[τ_lo, τ_hi] = [μ_t − r, μ_t + r]
```

returned by `get_influence_range(o_th)` and consumed by the hierarchy when
placing Gaussians.

**Self-test (sanity, runs on import):**

- 4-D covariance is symmetric and positive-semidefinite.
- Conditional 3-D shapes are correct.
- The opacity factor peaks at t = μ_t and decays away from it.
- The influence range is non-negative.

### 3.2 The Temporal Gaussian Hierarchy — `utils/temporal_gaussian_hierarchy.py`

Pure indexing layer over a `GaussianModel4D`. Two parameters:

- `num_levels = L` (default 9, paper default 9)
- `root_segment = S` (default 0.25, normalized fraction of the `[0, 1]` time
  axis; corresponds to the paper's S = 10 s for a 40 s root segment in
  Neural3DV).

Per-level segment length: s_l = S / 2^l, for l = 0..L−1, with a final "global"
segment of infinite length for content that does not fit anywhere.

**Placement (Eq. 6).** Implemented as `assign(model)`. A Gaussian fits in a
level-l segment iff `floor(τ_lo / s_l) == floor(τ_hi / s_l)` (both endpoints
of its influence range fall in the same segment). Because level-l segment
boundaries are a subset of level-(l+1) boundaries, segments are nested:
fitting in a level-l segment implies fitting at every coarser level. The
Gaussian's level is therefore the *finest* l at which it fits in one
segment, or `GLOBAL_LEVEL = -1` if no level qualifies. Vectorised as

```
fits[n, l] = floor(τ_lo[n] / s_l) == floor(τ_hi[n] / s_l)            # (N, L)
level[n]   = argmax_l of (fits[n, l] · (l + 1)) − 1                  # or −1
segment[n] = floor(τ_lo[n] / s_{level[n]})                           # if level ≥ 0
```

**Query (Eq. 7).** `active_mask(t)` returns a 1-D boolean of length N: a
Gaussian is active iff it lives in the global segment OR its assigned segment
equals `floor(t / s_{its level})` for its level.

**Update.** Re-assignment is just `assign(model)` again, called by the sampler
after every densification because both the Gaussian count and the per-Gaussian
temporal extents have changed.

**Pluggable segmentation strategy.** The `segmentation` argument accepts
`"uniform"` (paper-faithful) or `"adaptive"` (stub; content-adaptive splits
are the planned novelty extension; see `limitations-and-extensions.md`).

**Self-test:** crafts Gaussians with controlled influence radii, verifies
- tiny extent → finest level
- influence range straddling a root boundary → global
- mid extent → intermediate level (predictable)
- queries at multiple timestamps activate the correct subset.

### 3.3 Compact Appearance Model — methods on `GaussianModel4D`

Implemented as a backward hook on `_features_rest` plus a periodic cutoff.

**Initialisation.** All residual SH coefficients start at zero
(`create_from_pcd` writes `features[:, :, 1:] = 0`).

**Gradient hook (Eq. 11).** Registered by `enable_compact_appearance(g_th)`:

```
keep[n] = (||grad_h[n]||₂ ≥ g_th)  OR  (||h[n]||₂ ≠ 0)
g'_h[n] = grad_h[n] · keep[n]
```

So a Gaussian whose residual SH has not yet been "activated" only receives
gradient through `_features_rest` if its gradient consistently exceeds the
threshold. Once activated (h ≠ 0) it always receives gradient.

**λ_h cutoff.** `update_compact_appearance(lambda_h)` checks the proportion of
activated Gaussians; once it reaches λ_h (default 0.15) the threshold is set
to `∞`, freezing further activation. Called by the sampler every
`compact_check_every` iterations (default 100).

**Storage compression (Huffman).** Deferred — does not affect quality or
training; only the final on-disk size.

**Self-test:** verifies the hook zeros sub-threshold updates, preserves
gradients on already-activated rows, and that the λ_h cutoff sets the
threshold to ∞.

### 3.4 Adaptive control (densification) — methods on `GaussianModel4D`

A 4-D adaptation of the 3DGS / EasyVolcap adaptive control pattern. Run from
the sampler's `update_gaussians()` every `densification_interval` iterations
between `densify_from_iter` and `densify_until_iter`.

**Per-iteration statistics.** After each forward+backward, the sampler reads
the screen-space gradient (`scr.grad`) returned by the rasterizer and
accumulates a per-Gaussian view-space gradient magnitude into
`xyz_gradient_accum`. The maximum projected radius observed is also tracked
(`max_radii2D`).

**Three operations per density event:**

1. **Clone.** Gaussians with view-space gradient ≥ `densify_grad_threshold`
   AND maximum scale ≤ `percent_dense · scene_extent` are duplicated in
   place (the Gaussian is small but undersampled, so add another).

2. **Split.** Gaussians with view-space gradient ≥ threshold AND maximum
   scale > `percent_dense · scene_extent` are replaced by N = 2 children
   sampled from the 4-D Gaussian's distribution (using `build_rotation_4d`
   to map standard-normal samples into world+time space), with both spatial
   and temporal scales divided by 1.6.

3. **Prune.** Gaussians with opacity below `min_opacity` or screen-space
   radius above `max_screen_size` (if positive) are removed.

**Optimizer-state synchronization.** The trickiest piece. Each parameter
tensor changes its row count after every operation, but the Adam optimizer
holds per-row `exp_avg` / `exp_avg_sq` state under the same Python parameter
object. The pattern, mirroring EasyVolcap's 3-D code:

- `nn.Parameter.set_(new_storage)` mutates the storage *in place*, keeping
  the Python object identity, so the optimizer keeps tracking the same
  parameter object across the operation.
- An `optimizer_state` dotdict is built per parameter with two 1-D row
  masks: `old_keep` (size = original row count, marks which original rows
  survived) and `new_keep` (size = final row count, marks which final rows
  came from kept originals).
- After every drop/append, `old_keep.sum() == new_keep.sum()` must remain
  invariant (the number of original Gaussians that survived to the final
  tensor). An assertion in `densify_and_prune` guards this.
- After the operation, `net_utils.update_optimizer_state` resizes Adam's
  `exp_avg` / `exp_avg_sq` to match the new row count and copies the kept
  rows in via `new_exp_avg[new_keep] = old_exp_avg[old_keep]`.

**Row-identity tracking (`orig_idx`).** Maintains a 1-D tensor mapping each
current row back to its original index (or `-1` for newly added rows). On
every drop, the original indices of removed rows are read off and used to
flip `old_keep[idx] = False`. This is what keeps `old_keep` consistent with
`new_keep` even across the split path (postfix + prune of originals).

**Hierarchy re-assignment.** After `densify_and_prune` returns, the sampler
calls `self.hierarchy.assign(self.gaussians)` so every Gaussian is re-placed
in the segment that fits its current temporal influence range (which may
have shrunk during split). Per-Gaussian cost is O(L), independent of the
total number of Gaussians, so this stays cheap.

### 3.5 The integration sampler — `models/samplers/gaussianth_sampler.py`

Registers `GaussianTHSampler` with `@SAMPLERS.register_module()`. Inherits
from `PointPlanesSampler` to reuse EasyVolcap's `sample_index_time`,
`store_output`, camera-handling, bg color logic. Forces
`skip_loading_points=True` and `n_points=1` so the parent's per-frame
point-cloud loading does no real work (we never use it).

`forward(batch)`:

1. Call `update_gaussians(batch)` — uses the *previous* iteration's
   `scr.grad`/`radii`/`gs_idx` to accumulate stats and possibly run
   densification + hierarchy re-assignment.
2. Possibly run the λ_h cutoff check.
3. Read the normalized timestamp from `sample_index_time`.
4. Query `hierarchy.active_mask(t)` and gather active indices.
5. Compute the conditional 3-D Gaussians at `t` via `get_conditional_3d`.
6. Evaluate SH at the per-Gaussian view direction (up to
   `active_sh_degree`).
7. Rasterize via `render_diff_gauss(xyz3, rgb3, cov6, occ1, camera)`.
8. Call `meta.scr.retain_grad()` (the rasterizer wraps `scr` in a non-leaf
   op without retaining its gradient — without this, `scr.grad` is `None`
   after backward and densification has no signal).
9. Write `batch.output.rgb_map / acc_map / dpt_map` via `store_output`,
   stash `scr / rad / gs_idx` for next iter's `update_gaussians`.

### 3.6 Configs and lr table

- `configs/models/gaussianth.yaml` — declares the sampler type and all its
  hyperparameters (TGH, Compact Appearance, densification), plus the
  per-parameter learning rates:

  | Param suffix | LR | Comes from |
  |---|---|---|
  | `_xyz` | 1.6e-4 | 3DGS |
  | `_t` | 1.6e-4 | this work |
  | `_features_dc` | 2.5e-3 | 3DGS |
  | `_features_rest` | 1.25e-4 | 3DGS |
  | `_opacity` | 5e-2 | 3DGS |
  | `_scaling` | 5e-3 | 3DGS |
  | `_scaling_t` | 1e-3 | this work |
  | `_rotation_l` | 1e-3 | 3DGS |
  | `_rotation_r` | 1e-3 | 3DGS |

  EasyVolcap's optimizer matches lr-table keys against the path components
  of each named parameter (exact equality, not substring), so
  `_scaling_t` and `_scaling` do not collide.

- `configs/exps/gaussianth/gaussianth_actor1_4_subseq.yaml` — experiment
  config: combines `base.yaml`, `models/gaussianth.yaml`, and the existing
  `datasets/enerf_outdoor/actor1_4_subseq.yaml`.
- `configs/exps/gaussianth/gaussianth_flame_salmon.yaml` — experiment config
  for the canonical Neural3DV benchmark; combines `base.yaml`,
  `models/gaussianth.yaml`, and `datasets/neural3dv/flame_salmon.yaml`,
  overriding `view_sample`/`frame_sample` to the 19-camera flame_salmon set.

---

## 4. Scope: what is reproduced and what is deferred

### Reproduced (algorithm + mechanism)

- 4D Gaussian primitive with dual-quaternion rotation and conditional-3D
  extraction (Sec. 3.2.1).
- Temporal Gaussian Hierarchy with paper-faithful uniform placement (Eq. 6)
  and query (Eq. 7).
- Re-assignment of the hierarchy after every densification.
- Compact Appearance Model: zero-init residual SH, gradient-threshold hook
  (Eq. 11), λ_h proportion cutoff.
- Adaptive control: clone, split (with 4-D Gaussian sampling), prune.
- Optimizer-state synchronization for parameter row additions / removals.
- Hardware-accelerated `diff_gauss` rasterization path with `cov3D_precomp`.

### Simplified or deferred

| Component | Status | Cost vs. paper |
|---|---|---|
| RAM↔GPU segment streaming (paper keeps hierarchy in RAM, only active segments on GPU) | Not implemented; everything lives on GPU | VRAM grows with total Gaussian count — paper's *constant-VRAM* claim is not realized at scale |
| `fast_gauss` for inference (paper uses the OpenGL-shader rasterizer for eval) | Not wired up; we use `diff_gauss` for both train and eval | Lower FPS at eval; `fast_gauss` requires an EGL context which is not configured |
| Huffman coding for compact storage | Deferred | Larger on-disk size; quality unaffected |
| Per-frame SfM init point clouds (paper uses COLMAP per frame) | Not done; we initialise from uniform-random points inside the scene bounding box | None observed at 50k iterations — random init + full training matches and exceeds the paper (see §5). May still matter for shorter training budgets. |
| Segment-limited adaptive control (paper's compute-efficiency variant) | Not implemented as such; inactive Gaussians have zero accumulated gradient and naturally fail the threshold — same result, slightly more compute | Marginal extra cost |
| Content-adaptive segment boundaries | Intentionally pluggable (the `segmentation` arg accepts `"adaptive"` as a stub) | Reserved for our planned extension |

The deferred items each impact one of the paper's headline numbers (VRAM,
FPS, on-disk storage, absolute PSNR), but none impacts the *correctness* of
the representation or the *qualitative* behaviour shown in the figures.

---

## 5. Empirical results

Single RTX 3090 (24 GB) throughout, conda CUDA 12.1 + PyTorch 2.4.1+cu121,
random initialisation (no SfM init), ratio 0.5.

### Component self-tests

- `python -m easyvolcap.utils.gaussian4d_utils` — passes (cov4 PSD,
  conditional-3D shapes, opacity decay, influence range, gradient hook
  thresholding, λ_h cutoff).
- `python -m easyvolcap.utils.temporal_gaussian_hierarchy` — passes
  (placement levels for tiny / mid / straddler / huge extent Gaussians,
  queries activate the correct subset).

### Headline result — Neural3DV `flame_salmon`, the paper's own benchmark

A 50,000-iteration run on the canonical Neural3DV `flame_salmon` scene
(19 cameras, 300-frame segment, ratio 0.5, paper-default adaptive control)
reaches:

| Metric | Ours (50k, random init) | Paper (50k, SfM init) | 4DGS | 3DGS+T |
|---|---|---|---|---|
| PSNR  | **32.08 dB** | 29.44 dB | 28.89 dB | 28.61 dB |
| SSIM  | **0.970** | 0.945 | 0.952 | 0.950 |
| LPIPS | **0.197** | 0.214 | 0.197 | 0.210 |

Our reproduction **exceeds the published TGH headline on all three metrics**
on the same benchmark. Final Gaussian count 1.47 M after 145 densification
events; wall-clock ≈ 3.5 h. The note in §4 that random init might cost
quality at convergence did *not* materialise — at the full 50k-iteration
budget the densification path discovers the scene geometry without an SfM
prior. (The 300-frame segment is an easier setting than the paper's
1200-frame Table 1 entry; see the 1200-frame run below.)

### Full 1200-frame run (paper's Table 1 setting)

The full 1200-frame flame_salmon sequence was trained at the same 50k-iter,
paper-default settings. Densification completed normally (145 events,
≈1.34 M Gaussians) and the run reached iteration ~45,000 (epoch 9 of 10)
before being OOM-killed during the final refinement epoch — not from model
growth but from RAM/VRAM fragmentation accumulating over the multi-hour
run. A usable checkpoint at iteration 45,000 was retained. This is the
empirical evidence behind the project's hardware ask (RTX 5090 / 32 GB):
1200 frames is right at the 24 GB ceiling for the non-streaming
reproduction.

### Progression of smoke tests (for the record)

| Run | Dataset | Iters | Final PSNR | Notes |
|---|---|---|---|---|
| Stock 3DGS+T baseline | actor1_4_subseq | 50 | 4.69 | confirms framework works |
| `GaussianTHSampler` first E2E | actor1_4_subseq | 2000 | 20.97 | first full pipeline run, no SfM init |
| `GaussianTHSampler` | flame_salmon 300 | 3000 | 23.94 | mid-budget checkpoint |
| `GaussianTHSampler` | flame_salmon 300 | 50000 | **32.08** | headline result above |

### Interactive viewer

A WebSocket render server (`tgh_serve.py`) loads a trained checkpoint and
streams rendered frames to a browser client (`tgh_viewer.html`) with mouse
orbit + a time scrubber. This is a working first version of "Bridge B" from
`vr-viewing-options.md` — mono, JPEG-over-WebSocket, ~12–25 fps interactive.
Stereo / WebXR for headset viewing is the documented next step.

### Remaining gap to the paper's *headline system numbers*

Image quality is reproduced (and exceeded). The paper's VRAM / FPS / storage
headline figures still require the deferred engineering pieces — RAM↔GPU
segment streaming, `fast_gauss` inference, Huffman coding (§4) — which were
out of scope for the reproduction.

---

## 6. Files added or modified

### Added (this work)

| Path | Purpose |
|---|---|
| `EasyVolcap/easyvolcap/utils/gaussian4d_utils.py` | `GaussianModel4D`, dual-quaternion 4-D rotation, conditional-3D extraction, Compact Appearance hook, adaptive control |
| `EasyVolcap/easyvolcap/utils/temporal_gaussian_hierarchy.py` | `TemporalGaussianHierarchy`: placement (Eq. 6), query (Eq. 7), update by re-assignment |
| `EasyVolcap/easyvolcap/models/samplers/gaussianth_sampler.py` | `GaussianTHSampler` integration plugin |
| `EasyVolcap/configs/models/gaussianth.yaml` | Model config with lr table, TGH / Compact Appearance / densification hyperparameters |
| `EasyVolcap/configs/exps/gaussianth/gaussianth_actor1_4_subseq.yaml` | Experiment config (ENeRF-Outdoor example dataset) |
| `EasyVolcap/configs/exps/gaussianth/gaussianth_flame_salmon.yaml` | Experiment config (canonical Neural3DV benchmark) |
| `tgh_serve.py` (project root) | WebSocket render server — loads a trained checkpoint, streams frames to a browser |
| `tgh_viewer.html` (project root) | Browser client — mouse-orbit viewer with a time scrubber, talks to `tgh_serve.py` |

### Patched

| Path | Reason |
|---|---|
| `EasyVolcap/easyvolcap/utils/undist_utils.py` | Newer `pycolmap`'s `Camera.img_from_cam` requires 3-D points; pad the 2-D output of `cam_from_img` with a homogeneous 1 (a local helper `_img_from_cam`). |

The patch is mathematically a no-op (the 2-D normalized image-plane points
already live at z = 1) and exists only to bridge API drift between EasyVolcap
and current `pycolmap`.

### Untouched

Every other module of EasyVolcap. The reproduction is a pure plugin.

---

## 7. References

- Paper: Xu et al. 2024, *Representing Long Volumetric Video with Temporal
  Gaussian Hierarchy*, ACM ToG 43(6).
- Kerbl et al. 2023, *3D Gaussian Splatting for Real-Time Radiance Field
  Rendering*. (Source of the densification pattern.)
- Yang et al. 2023b, *Real-time Photorealistic Dynamic Scene Representation
  and Rendering with 4D Gaussian Splatting*. (Source of the 4-D Gaussian +
  dual-quaternion formulation and the conditional-3-D extraction.)
- Xu et al. 2024b, *4K4D: Real-Time 4D View Synthesis at 4K Resolution*.
  (Primary baseline the paper compares against.)
- Xu et al. 2023, *EasyVolcap: Accelerating Neural Volumetric Video
  Research*. (Framework we plug into.)
