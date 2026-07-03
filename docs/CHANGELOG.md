# Changelog

This file tracks project-level changes to the Long Volumetric Video / TGH
prototype. Dates use local project time.

## 2026-05-21

### Added

- Added active-subset conditioning for `GaussianTHSampler`.
  - The sampler now queries `TemporalGaussianHierarchy.active_indices(t)` before
    conditioning 4D Gaussians.
  - `GaussianModel4D.get_conditional_3d()` now accepts an optional `indices`
    argument and only builds covariance/conditional 3D splats for the selected
    rows.
  - SH feature lookup now supports selected rows through
    `GaussianModel4D.get_features_at(indices)`.
- Added a cached segment-index path in `TemporalGaussianHierarchy` so active
  rows can be gathered without scanning every Gaussian every iteration.
- Added a future-development concept section for a three-tier storage strategy:
  VRAM for active splats, RAM for nearby/inactive working sets, and SSD for cold
  temporal ranges.
- Added a fresh 50k benchmark launcher:
  `run_th_flame_salmon_50k_active_subset_benchmark.sh`.

### Changed

- The TGH training path now rasterizes only the active subset returned by the
  hierarchy.
- Densification statistics continue to scatter gradients back to global
  Gaussian rows through `gs_idx`, preserving the global optimizer state.
- `run_th_flame_salmon.sh` was normalized to LF line endings for WSL execution.
- Fixed the live viewer backend time mapping so a loaded checkpoint's sequence
  length updates `STATE.frames`, EasyVolcap's train/val `frame_sample`, and the
  sampler's own `frame_sample` together. This prevents a 300-frame checkpoint
  from being rendered through a stale 1200-frame timeline.
- Hardened the render server (`tgh_serve.py`) after a code review:
  - The server now binds `127.0.0.1` by default (new `--host` flag); the
    websocket `load` command validates that the requested checkpoint resolves
    under `CKPT_ROOT` (no absolute paths, no `..`/symlink escapes), since
    `torch.load(weights_only=False)` unpickles arbitrary files.
  - Checkpoint loads and renders run in worker threads behind a GPU lock so
    the event loop keeps answering keepalive pings during multi-second loads.
  - A checkpoint that leaves model keys unfilled now fails the `load` command
    instead of silently serving a half-loaded model.
  - `guess_frames` parses the `<N>f` token in checkpoint names instead of
    relying on a `1200` substring heuristic.
  - The flow-visualisation quantile subsamples above `2^24` elements to stay
    under the `torch.quantile` input size limit.

### Benchmark Notes

- A short 300-frame, 3k-iteration run completed successfully, but it is not
  quality-comparable to the previous 50k reference run.
- The 3k run produced substantially worse evaluation metrics because it stopped
  much earlier and ended with far fewer Gaussians:
  - 3k active-subset run: about 314k Gaussians.
  - Previous 50k reference: about 1.47M Gaussians.
- A fresh 300-frame, 50k active-subset benchmark completed successfully with
  comparable settings to the previous reference:
  - `exp_name=gaussianth_flame_salmon_active_subset_50k`
  - 50k iterations, 300 frames.
  - Densification from iteration 500 to 15,000.
  - 145 densification events.
  - Final post-densification count: about 1.48M Gaussians.
  - Wall-clock: 15:08:00 to 17:43:25, about 2h 35m.
  - Metrics: PSNR 31.9964, SSIM 0.97046, LPIPS 0.19674.

### Not Added Yet

- Differentiable hardware rasterization has not been integrated into the TGH
  training path yet. The current path uses the existing differentiable CUDA
  Gaussian rasterizer via `render_diff_gauss`.
- RAM/SSD streaming has not been implemented yet; it is documented as a future
  development direction.
