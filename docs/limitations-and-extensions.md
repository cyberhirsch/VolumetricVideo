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
| 1 | **Content-adaptive *temporal segment* boundaries for TGH** — replace the fixed `s_l = S/2^l` cuts with a motion-driven recursive split of the time axis (KD/BVH-style) | #3, #7 | ReCon-GS does dynamic hierarchy reconfiguration but for **anchor** positions, not for TGH's binary temporal partition; 4DGCPro does motion-aware *grouping* without changing TGH's cut rule | **Medium-high** — the specific algorithmic move (varying-length segment boundaries inside the TGH structure) is still open | High — pure algorithmic change, our `segmentation` arg already pluggable |
| 2 | **Differentiable hardware rasterization for 4D / dynamic scenes** — port the Yuan & He (May 2025) Vulkan programmable-blending backward pass from static 3DGS to 4D Gaussian Splatting with TGH-style temporal hierarchy | #1, #6 | **Yuan & He 2025** covers the *static-3DGS* case end-to-end. They explicitly leave 2DGS and mobile as future work; **4D / dynamic Gaussians are not addressed**. | **Low–medium** — engineering extension of an existing technique, not an algorithmic original contribution. Honest framing: *application paper*, not method paper. | Medium-hard — Vulkan/GLSL work + interaction with TGH active-subset streaming |
| 3 | **Online / streaming TGH** — append segments to the existing hierarchy as new frames arrive | new capability | ReCon-GS (anchor-based), Instant Gaussian Stream (per-frame) — both stream, neither in the TGH structure | **Medium** — engineering contribution rather than research novelty; still useful as a feature | High — natural fit |
| 4 | **Differentiable `fast_gauss` specifically** — backward pass for the dendenxu OpenGL geometry-shader rasterizer | #1, #6 | **Yuan & He 2025** covers the same idea in a *different graphics API* (Vulkan vs. OpenGL). The `fast_gauss`-specific OpenGL port would now be a *parallel implementation* of an already-published technique. | **Very low** — superseded except for niche OpenGL-only contexts | Medium-hard |
| 5 | **Inter-segment compression (P-frame-style residual coding)** | #5 | **4D-MoDe** (Sep 2025) does adaptive GOP keyframe insertion; 4DGCPro does hierarchical bitstream layering | **Low** — superseded | Medium |
| 6 | **Diffusion prior for sparse views** | #2 | **VDEGaussian** (Aug 2025) | **Low** — superseded | Hard — needs video diffusion + significant compute |

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
- Inter-segment compression and diffusion-sparse-view remain superseded
  by 4D-MoDe / VDEGaussian.

**Plan implication.** No change to the reproduction critical path. Once
the long training run is verified, the next research move is the
content-adaptive segmentation strategy. The Yuan-style 4D backward
remains attractive as a system contribution — useful for the
implementation, less useful as a publication anchor.
