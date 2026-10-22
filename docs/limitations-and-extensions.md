# Limitations & Extension Directions

Analysis of *"Representing Long Volumetric Video with Temporal Gaussian Hierarchy"*
(Xu et al., SIGGRAPH Asia 2024 / ACM ToG 43(6) Article 171), and where a follow-up
contribution could go.

---

## 1. Limitations the paper explicitly states (Section 7)

1. **Training is slow / not real-time.** Reconstruction "requires several hours to
   transform volumetric videos into our 4D representation" — 31.5 h for the 18,000-frame
   sequence. The paper's own suggested fix: instill a stronger geometry prior or
   regularize the distribution of Gaussian primitives.
2. **Needs semi-dense views.** The method "requires semi-dense views to adequately
   cover dynamic scenes, and it doesn't generalize well in a sparse-view setting."
   Suggested fix: use strong generative priors from diffusion models.

---

## 2. Weaknesses the paper does not foreground (inferred)

3. **Uniform segment boundaries.** The hierarchy subdivides time *uniformly* — segment
   lengths are exactly `S / 2^l` at fixed positions (Eq. 1, 5). A sudden event that
   straddles a boundary (e.g. a clap at t = 9.9 s near a 10 s root boundary) is forced
   into the global segment or split awkwardly. The paper notes the *count* of Gaussians
   per segment is non-uniform, but the *cut points* themselves are rigid.
4. **4D Gaussians cannot model abrupt motion / topology change.** A 4D Gaussian is a
   smooth spacetime ellipsoid — good for continuous motion, poor for sudden
   appearance/disappearance or fast non-rigid events.
5. **Constant VRAM, not constant storage.** Storage still grows with scene motion
   complexity (number of Gaussians). 2.2 GB for 18k frames is excellent but unbounded.
6. **No backward pass in the hardware rasterizer.** `fast_gauss` gives a 5x *rendering*
   speedup, but training still uses the slower `diff_gauss` software path.
7. **Fixed hyperparameters.** L = 9 levels and S = 10 s root segment are set empirically
   (Sec. 5.4), not adapted to scene content.
8. **Static-background, static-camera assumption.** The pipeline assumes a synchronized
   fixed camera array capturing a static background with a dynamic foreground.

---

## 3. Recent related work (2025–2026)

The TGH paper appeared December 2024; the field moved quickly afterwards and
several of the "obvious" extension directions now have entries in the
literature. The papers below all build on top of (or directly extend) 4D
Gaussian Splatting for dynamic / long volumetric video. PDFs of all of them
are mirrored in [`papers/`](papers/) next to this file.

A second pass in October 2026 added the entries marked *(Oct 2026 pass)*.
Two of them move the assessment in §4 and §5: SAGA, which already splits
time adaptively (for a codec), and ATGS, a direct long-sequence competitor
evaluated on the same benchmark as this reproduction.

### Long-sequence representation (direct competitors)

- **ATGS — *Anchored Temporal Gaussian Splatting for Long Volumetric Video
  Representation***. ACM ToG 45(4), Article 110 (SIGGRAPH 2026). arXiv
  [2608.30184](papers/ATGS_2026_arXiv2608.30184.pdf); code at
  [WuJH2001/ATGS](https://github.com/WuJH2001/ATGS). *(Oct 2026 pass)*
  Time-conditioned anchors initialised from keyframes, decoded through
  temporal feature grids within a sliding query window. Positions itself
  explicitly against TGH ("LongVolCap … degrades under complex and fast
  changing motions") and evaluates on **N3DV Flame Salmon** — the benchmark
  of this reproduction — plus VRU, MeetRoom and TGH's own SelfCap data.
  Worth noting for §5: its keyframes are **periodically sampled**, and their
  count is **set by hand per scene according to motion magnitude** (K = 15 on
  N3DV, K = 250 on VRU). The temporal granularity problem TGH has with fixed
  cuts is still being solved by manual tuning in 2026.

### Compression and storage

- **4D-MoDe — *Motion-Decoupled 4D Gaussian Compression for Editable and
  Scalable Volumetric Streaming***. arXiv [2509.17506](papers/4DMoDe_2025_arXiv2509.17506.pdf), Sep 2025.
  Layered representation that explicitly separates static background from
  dynamic foreground, with an **adaptive Group-of-Pictures (GOP) scheme**
  that inserts background keyframes only when needed. This is essentially
  the inter-segment / P-frame residual coding direction listed in §4 of this
  doc — done.

- **4DGCPro — *Efficient Hierarchical 4D Gaussian Compression for
  Progressive Volumetric Video Streaming***. arXiv
  [2509.17513](papers/4DGCPro_2025_arXiv2509.17513.pdf), Sep 2025.
  Motion-aware adaptive grouping, multi-level detail streaming, real-time
  mobile decoding, bandwidth-adaptive layer selection. Direct hierarchical
  successor of TGH for streaming.

- **MEGA — *Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes***
  (ICCV 2025). arXiv [2410.13613](papers/MEGA_Zhang2024_arXiv2410.13613.pdf).
  Claims **125× memory compression** vs vanilla 4DGS while matching speed and
  quality. Orthogonal to TGH (representation compression vs. temporal
  organisation) but closes much of TGH's headline VRAM/storage gap by other
  means.

- **PackUV — *Packed Gaussian UV Maps for 4D Volumetric Video***. arXiv
  [2602.23040](papers/PackUV_2026_arXiv2602.23040.pdf), Feb 2026. Packs
  Gaussians into video-codec-compatible UV atlases; scales to 30-min
  sequences; ships a 100-sequence / 2-billion-frame dataset.

- **SAGA — *Gaussian Splatting-based Volumetric Video Compression with
  Sparse 4D Anchors***. NeurIPS 2026. arXiv
  [2609.33969](papers/SAGA_2026_arXiv2609.33969.pdf). *(Oct 2026 pass)*
  A codec over a **hierarchical sparse 4D anchor scaffold**: candidate
  anchors are inserted into a quantised 4D Morton tree, nodes are split
  greedily by importance-weighted descriptor variance, and — the part that
  matters here — **temporal splitting is applied only where temporal
  variance is large**, leaving broad temporal support in static regions.
  This is content-adaptive subdivision of time, done for compression. The
  importance score it splits on combines normalised opacity, multi-view
  visibility, photometric residual and accumulated gradient magnitude.

- **QuARC-GS — *Quantized Anchored Residual Coding for Compact Dynamic
  Scene Streaming with Gaussian Splatting***. arXiv
  [2608.18285](papers/QuARCGS_2026_arXiv2608.18285.pdf), Aug 2026; code at
  [high-performance-computational-optics/QuARC-GS](https://github.com/high-performance-computational-optics/QuARC-GS).
  *(Oct 2026 pass)* Online free-viewpoint-video streaming for longer
  sequences via quantised residuals on anchors. Cites TGH as the
  long-sequence reference; another entry in the residual-coding direction
  (§4 #5).

### Streaming and online reconstruction

- **ReCon-GS — *Continuum-Preserved Gaussian Streaming for Fast and Compact
  Reconstruction of Dynamic Scenes***. NeurIPS 2025. arXiv
  [2509.24325](papers/ReConGS_2025_arXiv2509.24325.pdf).
  **Dynamic hierarchy reconfiguration with on-demand anchor
  re-hierarchization** and intra-hierarchical deformation inheritance.
  Anchor-based streaming hierarchy — overlaps substantially with the
  "content-adaptive hierarchy" idea, but applied to anchor positions rather
  than to TGH's temporal segment partition.

- **Instant Gaussian Stream**. arXiv
  [2503.16979](papers/InstantGaussianStream_2025_arXiv2503.16979.pdf),
  Mar 2025. **2.7 s per-frame** streaming reconstruction, **204 FPS** render.
  Per-frame online pipeline, not a hierarchy per se.

### Sparse-view and casual capture (TGH limitation #2)

- **VDEGaussian — *Video Diffusion Enhanced 4D Gaussian Splatting for
  Dynamic Urban Scenes Modeling***. arXiv
  [2508.02129](papers/VDEGaussian_2025_arXiv2508.02129.pdf), Aug 2025.
  Distils temporally consistent priors from a test-time-adapted video
  diffusion model — directly addresses the paper's sparse-view limitation.

- **4C4D — *4 Camera 4D Gaussian Splatting***. CVPR 2026. arXiv
  [2604.04063](papers/4C4D_2026_arXiv2604.04063.pdf). *(Oct 2026 pass)*
  High-fidelity 4DGS from **four portable cameras, without a diffusion
  prior**. The insight is that geometry, not appearance, is what fails
  under sparse views; a neural decaying function on Gaussian opacities
  shifts the gradients toward geometric learning. Far cheaper than the
  VDEGaussian route.

- **Sparse-View 4D Gaussian Splatting via Spatiotemporal Priors and
  Generative Assistance**. SIGGRAPH Asia 2026 Workshops. arXiv
  [2610.04606](papers/SparseView4DGS_SAVVC_2026_arXiv2610.04606.pdf).
  *(Oct 2026 pass)* Winning entry (30.04 dB full-frame PSNR) of the
  **Sparse-View Track of the SIGGRAPH Asia 2026 Volumetric Video
  Challenge** — six wide-baseline cameras. Mask-guided initialisation and
  densification plus spatiotemporal priors. Mainly useful as a pointer: the
  challenge is a ready-made public benchmark for sparse capture.

### Differentiable hardware rasterization (the direction we missed)

- **Yuan & He — *Efficient Differentiable Hardware Rasterization for 3D
  Gaussian Splatting***. arXiv
  [2505.18764](papers/EfficientDiffHWRasterizer_2025_arXiv2505.18764.pdf),
  May 2025. *Vulkan + GLSL* implementation of the backward pass via
  programmable blending plus a quad/subgroup hybrid gradient reduction
  strategy in the fragment shader. 3.07× end-to-end speedup vs. the CUDA
  tile-based 3DGS rasterizer; 4.14× memory reduction. Evaluated on static
  3DGS / MipNeRF360. Explicitly limited to static 3DGS in scope; future
  work section mentions 2DGS and mobile deployment, but not 4D / dynamic
  variants.

- **From Splats to Silicon — *Rethinking Computational Efficiency of
  3DGS***. arXiv [2609.06157](papers/SplatsToSilicon_2026_arXiv2609.06157.pdf),
  Sep 2026. *(Oct 2026 pass)* A systems survey and measurement study. Its
  hardware section covers the programmable-blending backward line (DHR),
  VR-Pipe, RayGS and NPU mappings — **none of it addresses 4D / dynamic
  Gaussians**, so §4 #2 stays open as of this pass. Ships **GSPROF**, an
  MIT-licensed profiling toolkit
  ([HAPPYPMN/gsprof](https://github.com/HAPPYPMN/gsprof)), directly usable for
  the benchmark-harness task.

### Pruning and importance (VRAM after densification)

All three target static 3DGS; listed because the "high VRAM after
densification" task needs a way to drop Gaussians without re-rendering
every view. Applying them to TGH means scoring per segment and handling the
time dimension. *(Oct 2026 pass)*

- **REFINE — *Super-efficient 3D Gaussian Splatting Pruning via
  Rendering-Free Primitive Importance***. arXiv
  [2606.09074](papers/REFINE_2026_arXiv2606.09074.pdf), Jun 2026. Importance
  from an analytically approximated, rendering-aware Hessian field — no
  rendering pass needed.
- **RAP — *Fast Feedforward Rendering-Free Attribute-Guided Primitive
  Importance Score Prediction for Efficient 3D Gaussian Splatting
  Processing***. CVPR 2026. arXiv
  [2602.19753](papers/RAP_2026_arXiv2602.19753.pdf); code at
  [yyyykf/RAP](https://github.com/yyyykf/RAP). Predicts importance from
  primitive attributes alone with a small feed-forward network.
- **NanoGS — *Training-Free Gaussian Splat Simplification***. ECCV 2026.
  arXiv [2603.16103](papers/NanoGS_2026_arXiv2603.16103.pdf).
  Simplification without retraining and without calibrated images.

### Web viewing

- **WebSplatter — *Enabling Cross-Device Efficient Gaussian Splatting in Web
  Browsers via WebGPU***. arXiv
  [2602.03207](papers/WebSplatter_2026_arXiv2602.03207.pdf), Feb 2026.
  *(Oct 2026 pass)* WebGPU rendering pipeline with a wait-free hierarchical
  radix sort that works around WebGPU's missing global atomics, plus
  opacity-aware culling before rasterization. Relevant to
  `tgh_viewer.html` if it moves to WebGPU.

### Background, for self-containment

- **3DGS** (Kerbl et al. 2023) — base rasterization framework. Not in
  `papers/` (already widely available).
- **4DGS / Yang et al. 2023b** — *Real-time Photorealistic Dynamic Scene
  Representation and Rendering with 4D Gaussian Splatting*. arXiv
  [2310.10642](papers/Yang2023_4DGS_arXiv2310.10642.pdf). The 4-D
  primitive + dual-quaternion + conditional-3-D formulation TGH builds on.
- **4D Gaussians (Wu et al. 2023b)** — arXiv
  [2310.08528](papers/Wu2024_4DGaussians_arXiv2310.08528.pdf). Alternative
  "deform-the-3DGS" branch the paper compares against.

---

## 4. Extension directions, re-ranked against the 2025–2026 landscape

The candidate directions, scored against (a) how much of the idea has
already been taken by the work above, (b) how cleanly it slots into our
existing `TemporalGaussianHierarchy` / `GaussianTHSampler` plugin, and
(c) tractability on a single RTX 3090.

| # | Direction | Addresses | Closest prior 2025–2026 work | Remaining novelty | Tractability on 3090 |
|---|---|---|---|---|---|
| 1 | **Content-adaptive *temporal segment* boundaries for TGH** — replace the fixed `s_l = S/2^l` cuts with a motion-driven recursive split of the time axis (KD/BVH-style) | #3, #7 | ReCon-GS does dynamic hierarchy reconfiguration but for **anchor** positions, not for TGH's binary temporal partition; 4DGCPro does motion-aware *grouping* without changing TGH's cut rule. *(Oct 2026)* **SAGA** splits a 4D Morton tree by variance and splits *in time* only where temporal variance is large — adaptive temporal subdivision, but of a **codec scaffold**, not of the hierarchy that decides which Gaussians are resident during training and rendering. **ATGS** still uses periodic keyframes with a hand-set count (K = 15 on N3DV, 250 on VRU). | **Medium** (was medium-high) — the move inside TGH's training-time hierarchy is still open, but SAGA has made variance-driven temporal splitting a published idea; the contribution must be framed around residency / VRAM, not representation | High — pure algorithmic change, our `segmentation` arg already pluggable |
| 2 | **Differentiable hardware rasterization for 4D / dynamic scenes** — port the Yuan & He (May 2025) Vulkan programmable-blending backward pass from static 3DGS to 4D Gaussian Splatting with TGH-style temporal hierarchy | #1, #6 | **Yuan & He 2025** covers the *static-3DGS* case end-to-end. They explicitly leave 2DGS and mobile as future work; **4D / dynamic Gaussians are not addressed**. *(Oct 2026)* The *From Splats to Silicon* survey's hardware section confirms nothing on the hardware path covers 4D yet. | **Low–medium** — engineering extension of an existing technique, not an algorithmic original contribution. Honest framing: *application paper*, not method paper. | Medium-hard — Vulkan/GLSL work + interaction with TGH active-subset streaming |
| 3 | **Online / streaming TGH** — append segments to the existing hierarchy as new frames arrive | new capability | ReCon-GS (anchor-based), Instant Gaussian Stream (per-frame) — both stream, neither in the TGH structure. *(Oct 2026)* QuARC-GS streams longer FVV with anchored residual coding — also outside the TGH structure. | **Medium** — engineering contribution rather than research novelty; still useful as a feature | High — natural fit |
| 4 | **Differentiable `fast_gauss` specifically** — backward pass for the dendenxu OpenGL geometry-shader rasterizer | #1, #6 | **Yuan & He 2025** covers the same idea in a *different graphics API* (Vulkan vs. OpenGL). The `fast_gauss`-specific OpenGL port would now be a *parallel implementation* of an already-published technique. | **Very low** — superseded except for niche OpenGL-only contexts | Medium-hard |
| 5 | **Inter-segment compression (P-frame-style residual coding)** | #5 | **4D-MoDe** (Sep 2025) does adaptive GOP keyframe insertion; 4DGCPro does hierarchical bitstream layering. *(Oct 2026)* SAGA (hierarchical 4D anchors with long-context entropy model), QuARC-GS (anchored residuals) | **Low** — superseded, more so after this pass | Medium |
| 6 | **Sparse-view 4D reconstruction** — diffusion or geometry prior | #2 | **VDEGaussian** (Aug 2025) via video diffusion. *(Oct 2026)* **4C4D** (CVPR 2026) reaches four cameras *without* diffusion, by biasing 4DGS toward geometry; the SIGGRAPH Asia 2026 Volumetric Video Challenge now has a six-camera sparse-view track | **Low** — superseded on both routes | **Medium** (was hard) — the 4C4D route needs no diffusion model, so it is now 3090-tractable as a baseline or integration; the challenge track is a ready benchmark |
| 7 | **Per-segment importance pruning to bound VRAM after densification** *(Oct 2026)* | #5, TASKS "high VRAM after densification" | REFINE (rendering-free Hessian importance), RAP (CVPR 2026, feed-forward importance from attributes), NanoGS (ECCV 2026, training-free) — all static 3DGS | **Low–medium** — adapting known pruning to 4D primitives and TGH segments; useful engineering, not a publication anchor | High — post-processing per segment, no change to training |

---

## 5. Recommended follow-up

**Primary candidate: content-adaptive *temporal segment* boundaries for
TGH.**

This is the direction that survived the literature re-check. The closest
2025–2026 work is ReCon-GS (anchor-position re-hierarchisation, not
temporal-segment re-cutting) and 4DGCPro (motion-aware *grouping* of
Gaussians *within* an existing temporal partition, not changing the
partition itself). The specific algorithmic move — replacing TGH's fixed
`s_l = S/2^l` cut points with a motion-driven recursive split of the
time axis — is **not directly taken** by any of the 2025/26 papers we
have. The contribution would be both *algorithmically clean* (a clear
ablation against the paper's own uniform baseline) and *technically
straightforward* (the `segmentation` argument on
`TemporalGaussianHierarchy` is already a plug-in slot waiting for an
`"adaptive"` implementation).

**October 2026 re-check.** The direction survives, with a narrower claim.
SAGA (NeurIPS 2026) already splits time adaptively — by temporal variance,
inside a 4D Morton tree — but for a compression scaffold over a trained
representation. What remains open is the adaptive partition of TGH's own
hierarchy, the structure that decides which Gaussians are resident while
training and rendering. Three consequences:

1. **Cite SAGA and frame the contribution around residency and VRAM**, not
   around "adaptive temporal splitting" as such — that phrase alone is now
   taken.
2. **SAGA's split score is a ready candidate criterion**: normalised
   opacity, multi-view visibility, photometric residual and accumulated
   gradient magnitude. It gives the adaptive strategy a defensible first
   version instead of an invented heuristic.
3. **ATGS becomes the baseline to beat.** It evaluates on Flame Salmon like
   this reproduction, its code is public, and its hand-set keyframe count
   (15 vs. 250 depending on scene motion) is the motivating example: a
   17× difference in temporal granularity that an adaptive scheme should
   find on its own.

**Secondary candidate: differentiable hardware rasterization extended to
4D / dynamic Gaussians.**

Honest framing: this is no longer an *algorithmic original contribution*
after Yuan & He 2025 — they did the hard part (per-pixel backward via
programmable blending + quad/subgroup gradient reduction) for static
3DGS. But the *extension to 4D Gaussian Splatting* (conditional-3D
extraction, TGH active-subset streaming) is not trivially derivable from
their paper and they themselves don't claim it. A combined system
(TGH + Yuan-style backward) would be a defensible *application /
integration paper*, not a method paper.

**De-prioritised:**

- The originally-planned differentiable `fast_gauss` is essentially
  superseded by Yuan & He — porting their idea from Vulkan to OpenGL is
  not a research contribution.
- Inter-segment compression remains superseded — by 4D-MoDe, and now also
  by SAGA and QuARC-GS.
- Sparse views remain superseded as research: VDEGaussian on the diffusion
  route, and now 4C4D (CVPR 2026) on a cheap non-diffusion route with four
  cameras. 4C4D is worth keeping as a baseline should sparse capture become
  a goal.
- Per-segment pruning (§4 #7) is worth doing for the VRAM problem, but as
  engineering rather than as a research anchor.

**Plan implication.** No change to the reproduction critical path. Once
the long training run is verified, the next research move is the
content-adaptive segmentation strategy, with ATGS added as a comparison
baseline. The Yuan-style 4D backward
remains attractive as a system contribution — useful for the
implementation, less useful as a publication anchor.
