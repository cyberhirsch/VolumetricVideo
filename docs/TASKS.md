# Task List

This file tracks implementation and benchmarking work for the Long Volumetric
Video / TGH prototype.

## In Progress

- [ ] Record final benchmark results in `docs/reproduction-methodology.md` and
  `README.md`.

## Next

- [ ] Validate active-subset conditioning numerically.
  - Compare `active_indices(t)` against the old `active_mask(t).nonzero()`.
  - Compare full conditioning plus slicing against direct indexed conditioning
    for several timestamps and checkpoints.
- [ ] Add a small benchmark harness.
  - Separate startup/data-loading time from training-loop time.
  - Record time to 5k, 10k, 15k, 20k, and final 50k.
  - Record peak VRAM and final checkpoint sizes.
- [ ] Investigate high VRAM after densification.
  - Confirm how much memory is parameters, Adam state, rasterizer buffers, and
    cached CUDA allocations.
  - Decide whether optimizer-state compression or staged updates are needed.
- [ ] Add regression tests for the TGH active-set path.
  - Segment index rebuild after `assign()`.
  - Cache invalidation after checkpoint load.
  - Gradient scatter through `gs_idx` after indexed rendering.

## Future Development

- [ ] Design RAM/SSD streaming for long videos.
  - VRAM: active segments and immediate working buffers.
  - RAM: nearby segments and recently used inactive splats.
  - SSD: cold temporal segments and static/background partitions not needed for
    the current training window.
- [ ] Prototype streaming at checkpoint/export level before integrating it into
  the live training loop.
- [ ] Evaluate differentiable hardware rasterization options.
  - Current TGH path uses `render_diff_gauss`.
  - Candidate paths include existing repo OpenGL/nvdiffrast utilities or a
    dedicated Gaussian hardware raster path.
  - Must preserve gradients required for position, opacity, covariance, SH, and
    densification statistics.
- [ ] Explore quality/speed tradeoffs after the active-subset benchmark.
  - Densification schedule.
  - SH activation schedule.
  - Active/global segment balance.
  - Optional lower-precision optimizer state.

## Done

- [x] Fix live viewer/backend time mapping for loaded checkpoints.
  - `tgh_serve.py` now applies the selected frame count to `STATE.frames`,
    EasyVolcap train/val `frame_sample`, and `MODEL.sampler.frame_sample`.
  - Sanity check: frame 150 on the 300-frame active-subset checkpoint maps to
    sampler time `t=0.501672` and sampler index `150`.
- [x] Finish the fresh 50k active-subset benchmark.
  - Experiment: `gaussianth_flame_salmon_active_subset_50k`.
  - Exit code: 0.
  - Wall-clock: 15:08:00 to 17:43:25, about 2h 35m.
  - Metrics: PSNR 31.9964, SSIM 0.97046, LPIPS 0.19674.
  - Densification events: 145.
  - Final post-densification count: about 1.48M Gaussians.
- [x] Update `docs/CHANGELOG.md` with the final active-subset benchmark result.
- [x] Implement active-subset conditioning in `GaussianTHSampler`.
- [x] Add indexed covariance and conditional 3D Gaussian computation in
  `GaussianModel4D`.
- [x] Add cached active-index lookup in `TemporalGaussianHierarchy`.
- [x] Document RAM/VRAM/SSD streaming as a future concept.
- [x] Run a 300-frame smoke benchmark to confirm the active-subset path trains.
