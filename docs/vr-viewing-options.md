# Viewing the TGH Splat in VR

How to bring our trained Temporal Gaussian Hierarchy (TGH) model into a VR
headset, ranked from "this week" to "real engineering project."

The catch up front: **4D Gaussian Splats in VR is genuinely rare in 2026.**
Most VR splat viewers are *static-3D-only*. None natively understands the TGH
on-disk format. Every viable path therefore goes through one of three bridges
between our trained model and an existing VR display pipeline.

---

## 1. Tooling landscape (May 2026)

| Tool | Static 3DGS | Animated / 4D | VR support | License | Notes |
|---|---|---|---|---|---|
| **[Splatapult](https://github.com/hyperlogic/splatapult)** | ✓ | per-frame `.ply` sequence | OpenXR (Quest, Index, Vive) | MIT | Single binary, has `--sequence` mode for animated splats |
| **[aras-p Unity Gaussian Splatting](https://github.com/aras-p/UnityGaussianSplatting)** | ✓ | scriptable | Unity → OpenXR | MIT | Lots of customization, deploys to Quest standalone |
| **[PlayCanvas SuperSplat](https://playcanvas.com/supersplat)** | ✓ | per-frame sequence | WebXR | Apache 2.0 | Browser-native, no install for viewers |
| **[gsplat.js (antimatter15)](https://github.com/antimatter15/splat)** | ✓ | manual sequence loading | WebXR | MIT | Tiny WebGL viewer, ~500 lines |
| **[mkkellogg gaussian-splats-3d](https://github.com/mkkellogg/GaussianSplats3D)** | ✓ | yes (animation API) | WebXR | MIT | Has a documented animated-sequence example |
| **Spectacular AI 3DGS Viewer** (Quest Store) | ✓ | no | Quest standalone | Free / proprietary | One-tap install, easiest setup but static only |
| **Postshot** | ✓ | limited | Desktop VR | Commercial | Pipeline tool, not just viewer |
| **NerfStudio gsplat + viser** | ✓ | partial | no native VR | Apache 2.0 | Web viewer only |
| **EasyVolcap GUI** (`evc-gui`) | TGH native | yes | **no VR** | MIT | What we have — desktop OpenGL only |

The two missing pieces in the matrix are:
- *TGH-native* viewers (only EasyVolcap, and it's not VR), and
- *4D-aware* VR viewers (none — every "animated" path is per-frame `.ply` swapping).

Every option below bridges those gaps differently.

---

## 2. Bridge A — Bake per-frame `.ply`, play in a standard viewer

**Idea.** Use our `GaussianModel4D.get_conditional_3d(t)` to materialise the 3-D
Gaussian set at every frame, write each set as a standard 3DGS `.ply`, and
hand the resulting sequence to any animated-splat viewer.

### Pipeline

```
TGH checkpoint (.pt)
   │
   │  for t in [0..N-1]:
   │      mask = hierarchy.active_mask(t / (N-1))
   │      xyz3, cov6, occ1 = gaussians.get_conditional_3d(t / (N-1))
   │      write_ply(frame_NNN.ply,
   │                xyz=xyz3[mask],
   │                f_dc/f_rest=features[mask],
   │                opacity=occ1[mask],
   │                scale/rot=from cov6[mask],   # or store cov directly
   │                normal=zeros)
   ▼
~/lvv-data/result/.../ply/frame_000.ply
                       frame_001.ply
                       ... × N
   │
   ▼  load into:
       Splatapult --sequence ply/    (Quest / Index, OpenXR)
   or  PlayCanvas SuperSplat         (WebXR, drag-drop)
   or  Unity aras-p plugin           (custom C# loader, OpenXR)
```

### Format notes

The standard 3DGS `.ply` layout (compatible with every tool above) has these
per-vertex attributes:

```
x, y, z              # spatial mean
nx, ny, nz           # normal (zeros, ignored by 3DGS)
f_dc_{0..2}          # DC SH band (3 values, = base color)
f_rest_{0..(N-1)*3}  # higher SH bands ((deg+1)^2-1)*3 values
opacity              # logit-space opacity (the raw _opacity, pre-sigmoid)
scale_0, scale_1, scale_2   # log-space spatial scales
rot_0, rot_1, rot_2, rot_3  # quaternion, unnormalised
```

Two pieces of glue are needed when exporting from a 4-D Gaussian:

1. **From the conditional 3-D covariance back to (scale, rotation).** Standard
   3DGS expects `(scale, rot)`, not a `cov3` precomp. The recipe is an
   eigendecomposition of the symmetric `cov3` (3×3): the eigenvalues are
   `scale²` (take √, then log for storage), and the eigenvectors form the
   rotation matrix (then convert to quaternion).
2. **Per-Gaussian opacity already has the temporal marginal factor baked in**
   thanks to `get_conditional_3d` — store the *logit* of that conditioned
   opacity so the viewer's sigmoid reproduces the correct alpha.

### Expected footprint

- Per-frame Gaussian count: depends on `hierarchy.active_mask(t).sum()`. For
  our 300-frame `flame_salmon` run with ~313 k total Gaussians, the active
  subset is ~50–80 k per frame (one segment per level + global).
- `.ply` size per frame at 65 k Gaussians, SH degree 3: ≈ 18 MB.
- 300 frames × 18 MB ≈ **5.4 GB** on disk.
- Loaded in a Quest 3 (12 GB RAM): comfortable. Older Quest 2 (6 GB): tight,
  may need to subsample.

### Trade-offs

| Pro | Con |
|---|---|
| Works with mature, free tools today | **Loses the constant-memory property of TGH at viewing time** — that's literally what TGH was designed to avoid, but for a viewing-only baking step we accept it |
| No client-side custom code | 5.4 GB on the headset / streaming target |
| Compatible with any 3DGS viewer | No view-dependent SH evaluation in some lightweight viewers (color may look slightly different) |
| Easy to share / embed | Per-frame loading time at swap (~10-30 ms — may cause hitching at high FPS unless preloaded) |

### Effort

A `scripts/export_tgh_to_ply_sequence.py` writer is ~40 lines of Python on top
of the existing `GaussianModel4D` / `TemporalGaussianHierarchy` interface.
The hardest part is the cov → (scale, rot) decomposition (~10 lines with
`torch.linalg.eigh`).

### Recommended viewer for this bridge

**Splatapult.** Free, single binary, runs on Linux/Mac/Windows + Quest via Link
or AirLink, has an explicit `--sequence` mode for animated splats. Fewest moving
parts.

---

## 3. Bridge B — WebSocket-streaming TGH + WebXR client

**Idea.** Run the TGH model on the host as a render server; the headset
displays whatever the server renders in real time over the network. Preserves
TGH's *constant-memory* property because rendering still happens through
`GaussianTHSampler.forward`, which only touches the active segment subset.

### Pipeline

```
Headset (Quest browser / WebXR app)
   │  WebXR pose updates: 90 Hz, stereo
   ▼
WebSocket  (already exists in EasyVolcap: `evc-ws`)
   ▲
Host (RTX 3090)
   │  per eye, per frame:
   │    set camera = WebXR pose
   │    GaussianTHSampler.forward(t)  -> JPEG/H.264 frame
   │    send back
   ▼
```

### What EasyVolcap already provides

- `easyvolcap/runners/websocket_server.py` — render-on-demand over WebSocket
- `easyvolcap/runners/volumetric_video_viewer.py` — the OpenGL desktop viewer
  this protocol mirrors

### What needs to be added

1. **A WebXR client** (HTML + a few hundred lines of JS) that
   - establishes a WebXR Immersive session,
   - sends the per-frame stereo camera pose over WebSocket,
   - decodes the two JPEGs/H.264-frames the server returns,
   - blits them into the left/right framebuffers each frame.
2. **Stereo rendering in the server** — the existing `evc-ws` renders one
   monocular frame per request; need to either send the pose as a pair or call
   `forward()` twice per frame.
3. **Latency hiding** — at the typical Quest-Link ~30-60 ms round-trip,
   raw "render-then-display" causes head lag. Mitigations:
   - **Asynchronous Time Warp** on the client: the WebXR runtime already does
     this, but only against the *most recent* received frame.
   - **Predictive pose**: send pose extrapolated to the expected display time.
   - **Foveated streaming**: send a high-res central tile + low-res
     periphery.

### Trade-offs

| Pro | Con |
|---|---|
| Full TGH benefits at view time (constant memory, hierarchy queries) | Network latency — Quest standalone over Wi-Fi 6 ≈ 20-40 ms; tethered ≈ 10-20 ms |
| Server can be a beefy desktop; client is just a browser | Requires reliable network between host and headset |
| Basis for a "TGH for telepresence" use case — meaningful immersive-media application | Real engineering — stereo, time warp, codec |
| Works on Quest standalone (no Link cable) via the built-in browser | First-time setup: WebXR session, ws endpoint, codec choice |

### Effort

A few solid days of work. The pieces exist (EasyVolcap's ws server, the
Babylon.js / Three.js WebXR API, hardware video decoding in the browser); the
work is the integration plus latency tuning.

### Why this is interesting beyond viewing

This is also the underlying technical pattern for the "**TGH for live
volumetric telepresence**" story — exactly the kind of application a professor
of immersive media might be interested in for a follow-up paper. The same
infrastructure that lets you view a recording in VR is what would let you
stream a live capture into the headset of a remote viewer with constant
bandwidth regardless of session length.

---

## 4. Bridge C — Native VR via a Unity / OpenXR port of `fast_gauss`

**Idea.** The TGH paper's whole point of `fast_gauss` is that the GPU's
fixed-function rasterization pipeline already exists for this exact operation
(rasterize many small primitives with alpha blending). Re-target that pipeline
for stereo VR and run the whole rendering loop on the headset GPU (or on a
desktop GPU and submit to OpenXR).

### What's involved

1. **Port the OpenGL geometry shader to Unity HLSL** (Universal Render Pipeline
   or Built-in). The shader logic is short — billboard each Gaussian, alpha
   blend in a sorted order — but Unity's URP doesn't expose GS cleanly. May
   need either:
   - a custom render feature in URP, or
   - a vertex-shader-only "expand four corners" trick (no GS needed), which is
     in fact how some web viewers implement gsplat.
2. **CUDA-side sorting** — the depth sort that `fast_gauss` does on the GPU
   would need a Unity-friendly replacement. Compute shaders can do radix sort
   on the GPU, but at considerably worse throughput than CUDA.
3. **TGH active-set query on Unity side** — feed the active Gaussian set per
   frame from a C#-side hierarchy implementation, or stream it from the host.
4. **Two-eye render** — straightforward in OpenXR / Unity XR Plugin
   Management.

### Trade-offs

| Pro | Con |
|---|---|
| Native 90/120 Hz stereo, no streaming | **Weeks of engineering** |
| Could ship as a Quest standalone app | Loses GPU-CUDA-radix-sort throughput unless we wire CUDA-Unity interop (only available on Windows + RTX with native plugin) |
| Best possible quality + lowest latency | One platform per binary |

### When this is worth doing

Only if this becomes the actual deliverable of the project — e.g. a
"Real-time 4D Gaussian Splatting in VR via Hardware-Rasterized TGH Streaming"
publication. As a pure viewer, Bridge A is 1000× cheaper for nearly the same
viewing experience.

---

## 5. Recommendation by goal

| Your goal | Best bridge | Wall-clock to first VR view |
|---|---|---|
| "See my model in VR by tonight." | **A — Splatapult + per-frame `.ply`** | ~3-4 hours total (export script + Splatapult setup + headset session) |
| "Show off the TGH constant-memory property in a live demo." | **B — WebSocket + WebXR** | 3-5 days of integration |
| "Ship a Quest app / publish a real-time VR paper." | **C — Native Unity port** | Multi-week engineering |
| "I just want to verify the trained model looks right." | None — use `evc-gui` on desktop | Minutes (already trained) |

---

## 6. Specifically for *our* trained model

We currently have at `~/lvv-data/trained_model/gaussianth_flame_salmon/latest.pt`:

- 313,460 4D Gaussians, level distribution roughly `[40k, 80k, 100k, 50k,
  10k, …, global]`.
- Active subset per frame: ~50-80 k (one segment per level + global).
- Frame count: 300, FPS 30 → 10 seconds of volumetric video.

**Concrete first step we can take this session:**

A `scripts/export_tgh_to_ply_sequence.py` that loads the checkpoint and writes
`~/lvv-data/result/gaussianth_flame_salmon/ply/frame_NNN.ply` for N in
`[0..299]`. The script reuses `GaussianModel4D.get_conditional_3d` and
`TemporalGaussianHierarchy.active_mask` so there is no behavior duplication
relative to what the renderer does at training time.

Then on the headset side, you install Splatapult (one binary download) and
point it at the `ply/` directory. That ships you a stereo VR walk-through of
the salmon-cooking scene by end of day.

If the desktop-VR experience is too easy and you want the TGH property
visible at view time, that's when Bridge B becomes the next step.
