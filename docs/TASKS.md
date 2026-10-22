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
  - Candidate tool: GSPROF (MIT, https://github.com/HAPPYPMN/gsprof, from
    *From Splats to Silicon*, arXiv 2609.06157) — per-call timings and peak
    allocated memory including constructed buffers and temporaries, which is
    the breakdown the VRAM task below needs.
- [ ] Investigate high VRAM after densification.
  - Confirm how much memory is parameters, Adam state, rasterizer buffers, and
    cached CUDA allocations.
  - Decide whether optimizer-state compression or staged updates are needed.
  - Try change-gated densification (QuARC-GS §3.5, arXiv 2608.18285): only
    densify where content genuinely changes over time, not in static regions
    that merely have high error. Per Gaussian,
    `g_i = ‖∇_{x_i} (L1(Î, I_t) − L1(Î, I_{t−1}))‖₂`, normalised to
    `w_i = g_i / (g_i + median(g))`; multiply the accumulated densification
    gradient by `w_i` before clone/split. No second render — one extra
    backward with a different image gradient, on a subset of views.
  - Gate only the finer temporal segments: the root/global segment carries the
    static background, which this rule would otherwise stop densifying.
  - Measure against the 50k baseline (1.48M Gaussians after 145
    densification events).
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
  - Reference for the unit of residency: ATGS §3.1–3.3 (ToG 2026, arXiv
    2608.30184). Training updates only the anchors and temporal grids of the
    current timestamp; rendering uses a fixed window of W = 7 keyframes; each
    temporal grid is a bounded hash table (2^16 entries). The working set is
    known and bounded in advance — the property the VRAM/RAM/SSD tiers need.
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

- [x] Harden `tgh_serve.py` after code review.
  - Bind `127.0.0.1` by default (`--host` to override); reject `load` paths
    that escape `CKPT_ROOT` (torch.load unpickles, so client paths are unsafe).
  - Run checkpoint loads/renders in worker threads behind a GPU lock so
    keepalive pings keep flowing during long loads.
  - Fail `load` when model keys are left unfilled; parse `<N>f` from names in
    `guess_frames`; subsample the flow quantile above the torch.quantile limit.
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
