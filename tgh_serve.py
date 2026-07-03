#!/usr/bin/env python
"""TGH WebSocket render server.

Loads the trained GaussianTHSampler checkpoint and serves rendered frames
over WebSocket to a browser client (see tgh_viewer.html).

Protocol:
  Client -> Server (JSON):  { K, R, T, H, W, t, mode }
    K: 3x3 intrinsics (list of lists)
    R: 3x3 world-to-camera rotation (list of lists)
    T: 3x1 world-to-camera translation (list of lists)
    H, W: image height/width
    t: normalized time in [0, 1]
    mode: "rgb" (default) | "flow" | "blend"
      rgb   - the ordinary rendered colour image
      flow  - per-Gaussian motion visualised with the Middlebury colour wheel
      blend - the colour-coded motion glow over a dimmed greyscale scene

  Client -> Server (JSON):  { cmd: "load"|"set_frames"|"info", ... }
    load        { cmd:"load", ckpt:"<dir>/<file>.pt", frames?:int }
    set_frames  { cmd:"set_frames", frames:int }
    info        { cmd:"info" }   -> request the info message below

  Server -> Client (binary):  JPEG bytes of a rendered frame.
  Server -> Client (JSON):    { type:"info", checkpoints, ckpt, frames, fps,
                                gaussians }  - sent on connect and after load.

Run as:
  bash serve.sh
or directly:
  python tgh_serve.py [--host 127.0.0.1] [--port 8765] [--ckpt <dir>/latest.pt]
                      [--frames N]
"""

import os
import re
import sys
import io
import json
import math
import glob
import argparse
import asyncio
import time
from pathlib import Path

# Inject argv to look like an EasyVolcap test invocation BEFORE importing
# anything from easyvolcap — the engine reads sys.argv at import time.
_EVC_DIR = "/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
os.chdir(_EVC_DIR)

_CLI_ARGV = sys.argv[1:]
sys.argv = [
    "tgh_serve.py",
    "-t", "test",
    "-c", "configs/exps/gaussianth/gaussianth_flame_salmon.yaml",
    # full 1200-frame timeline accessible at eval-time
    "val_dataloader_cfg.dataset_cfg.frame_sample=[0,1200,1]",
    "val_dataloader_cfg.dataset_cfg.ratio=0.5",
    "dataloader_cfg.dataset_cfg.frame_sample=[0,1200,1]",
    "dataloader_cfg.dataset_cfg.ratio=0.5",
    "val_dataloader_cfg.dataset_cfg.dataloading_workers=4",
    "dataloader_cfg.dataset_cfg.dataloading_workers=4",
    "model_cfg.sampler_cfg.skip_loading_points=True",
]

import torch
import numpy as np
from PIL import Image
import websockets

from easyvolcap.engine import cfg, MODELS, DATALOADERS, RUNNERS
from easyvolcap.utils.import_utils import discover_modules
discover_modules()  # populate the registries by importing every submodule
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.console_utils import log
from easyvolcap.utils.data_utils import to_x
from easyvolcap.utils.gaussian_utils import render_diff_gauss, prepare_gaussian_camera


# --------------------------------------------------------------- server state
CKPT_ROOT = os.path.expanduser("~/lvv-data/trained_model")

# Mutable globals shared with the websocket handlers.
MODEL = None
STATE = dotdict(ckpt=None, frames=1200, fps=30, gaussians=0)

# Serializes GPU work (checkpoint loads, renders) that runs in worker threads
# so the event loop stays free to answer pings while the GPU is busy.
GPU_LOCK = asyncio.Lock()


def list_checkpoints():
    """Every '<dir>/<file>.pt' under CKPT_ROOT, latest.pt first per directory."""
    rels = []
    for pt in glob.glob(os.path.join(CKPT_ROOT, "*", "*.pt")):
        rels.append(os.path.relpath(pt, CKPT_ROOT).replace(os.sep, "/"))
    rels.sort(key=lambda p: (os.path.dirname(p), not p.endswith("latest.pt"), p))
    return rels


def guess_frames(rel: str) -> int:
    """Best-effort sequence length from a checkpoint name (overridable).

    Checkpoint files are named like '..._1200f_latest.pt'; the '<N>f' token is
    authoritative. Fall back to the old substring heuristic for older names.
    """
    m = re.search(r"(\d+)f", rel.lower())
    if m:
        return int(m.group(1))
    return 1200 if "1200" in rel.lower() else 300


def build_model():
    """Construct the VolumetricVideoModel (no dataloader / runner).

    A render server does not need a val_dataloader, which would otherwise
    spend 5+ minutes decoding the whole training set into memory.
    """
    return MODELS.build(cfg.model_cfg).cuda()


def set_sequence_frames(frames: int):
    """Apply the playback/training sequence length to every time-mapping path."""
    frames = max(1, int(frames))
    frame_sample = [0, frames, 1]

    STATE.frames = frames
    # sample_index_time reads the val dataset frame_sample and the sampler's
    # own frame_sample at eval time; the train dataloader is never consulted
    # by the render server.
    cfg.val_dataloader_cfg.dataset_cfg.frame_sample = frame_sample.copy()

    if MODEL is not None and hasattr(MODEL, "sampler"):
        sampler = MODEL.sampler
        sampler.frame_sample = frame_sample.copy()
        if hasattr(sampler, "n_frames"):
            sampler.n_frames = frames

    log(f"sequence frames set to {frames}, frame_sample={frame_sample}")


def resolve_checkpoint_path(name: str) -> str:
    """Resolve a client-supplied checkpoint name to a real file under CKPT_ROOT.

    torch.load unpickles (weights_only=False), so a client must never be able
    to point the server at an arbitrary file: reject absolute paths and any
    relative path that escapes CKPT_ROOT via '..' or symlinks.
    """
    root = os.path.realpath(CKPT_ROOT)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath([path, root]) != root:
        raise ValueError(f"checkpoint path escapes checkpoint root: {name}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"checkpoint not found: {name}")
    return path


def load_checkpoint(name: str, frames: int = None, trusted: bool = False):
    """Load a checkpoint into the global MODEL and update STATE.

    `name` is a '<dir>/<file>.pt' path relative to CKPT_ROOT (or, for
    trusted=True callers like the CLI, an absolute path). `frames` overrides
    the auto-guessed sequence length.
    """
    if trusted and os.path.isabs(name):
        path = name
        if not os.path.isfile(path):
            raise FileNotFoundError(f"checkpoint not found: {path}")
    else:
        path = resolve_checkpoint_path(name)
    rel = os.path.relpath(path, CKPT_ROOT).replace(os.sep, "/")

    log(f"loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    state = ckpt.get("network", ckpt.get("model", ckpt))
    res = MODEL.load_state_dict(state, strict=False, assign=True)
    if res.unexpected_keys:
        log(f"unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:5]} ...")
    if res.missing_keys:
        log(f"missing keys ({len(res.missing_keys)}): {res.missing_keys[:5]} ...")
        raise RuntimeError(
            f"checkpoint {rel} left {len(res.missing_keys)} model keys "
            f"unfilled; refusing to serve a half-loaded model")
    MODEL.eval()

    # The hierarchy caches a segment index built against the PREVIOUS Gaussian
    # set; drop it so active_indices() rebuilds against the freshly loaded
    # buffers (otherwise a reload renders the wrong active subset).
    h = MODEL.sampler.hierarchy
    h._segment_index = None
    h._global_indices = None

    STATE.ckpt = rel
    set_sequence_frames(int(frames) if frames else guess_frames(rel))
    STATE.gaussians = len(MODEL.sampler.gaussians)
    log(f"loaded {rel}: {STATE.gaussians} Gaussians, "
        f"frames={STATE.frames}, fps={STATE.fps}")


def build_render_batch(req: dict) -> dotdict:
    """Build the batch dict that GaussianTHSampler.forward expects."""
    H, W = int(req["H"]), int(req["W"])
    K = np.asarray(req["K"], dtype=np.float32)
    R = np.asarray(req["R"], dtype=np.float32)
    T = np.asarray(req["T"], dtype=np.float32)
    if T.ndim == 1:
        T = T[:, None]

    t_norm = float(req["t"])
    # Map normalized t to a frame index; sample_index_time will then recompute
    # t internally during eval mode.
    n_train_frames = STATE.frames
    latent_index = int(round(t_norm * (n_train_frames - 1)))
    latent_index = max(0, min(n_train_frames - 1, latent_index))

    near = float(req.get("n", 0.5))
    far = float(req.get("f", 200.0))

    def t_cpu(x): return torch.as_tensor(x)
    def t_gpu(x): return torch.as_tensor(x).cuda()

    batch = dotdict()
    batch.H = t_gpu([H])
    batch.W = t_gpu([W])
    batch.K = t_gpu(K).unsqueeze(0)
    batch.R = t_gpu(R).unsqueeze(0)
    batch.T = t_gpu(T).unsqueeze(0)
    batch.n = t_gpu([near])
    batch.f = t_gpu([far])
    batch.t = t_gpu([t_norm])
    batch.latent_index = t_gpu([latent_index])

    batch.meta = dotdict()
    batch.meta.H = t_cpu([H])
    batch.meta.W = t_cpu([W])
    batch.meta.K = t_cpu(K).unsqueeze(0)
    batch.meta.R = t_cpu(R).unsqueeze(0)
    batch.meta.T = t_cpu(T).unsqueeze(0)
    batch.meta.n = t_cpu([near])
    batch.meta.f = t_cpu([far])
    batch.meta.t = t_cpu([t_norm])
    batch.meta.latent_index = t_cpu([latent_index])
    batch.meta.iter = 0

    batch.output = dotdict()
    return batch


# --------------------------------------------------------------- motion colours
_COLOR_WHEEL = None


def _color_wheel() -> torch.Tensor:
    """The standard Middlebury optical-flow colour wheel as a (55, 3) CUDA tensor.

    Hue encodes flow direction; the magnitude-blend toward white is applied
    later in `flow_uv_to_rgb`.
    """
    global _COLOR_WHEEL
    if _COLOR_WHEEL is not None:
        return _COLOR_WHEEL
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    w = np.zeros((ncols, 3), dtype=np.float32)
    c = 0
    w[c:c+RY, 0] = 1.0;                    w[c:c+RY, 1] = np.arange(RY) / RY;  c += RY
    w[c:c+YG, 0] = 1.0 - np.arange(YG)/YG; w[c:c+YG, 1] = 1.0;                 c += YG
    w[c:c+GC, 1] = 1.0;                    w[c:c+GC, 2] = np.arange(GC) / GC;  c += GC
    w[c:c+CB, 1] = 1.0 - np.arange(CB)/CB; w[c:c+CB, 2] = 1.0;                 c += CB
    w[c:c+BM, 2] = 1.0;                    w[c:c+BM, 0] = np.arange(BM) / BM;  c += BM
    w[c:c+MR, 2] = 1.0 - np.arange(MR)/MR; w[c:c+MR, 0] = 1.0
    _COLOR_WHEEL = torch.from_numpy(w).cuda()
    return _COLOR_WHEEL


def flow_uv_to_rgb(fx: torch.Tensor, fy: torch.Tensor,
                   max_mag: torch.Tensor, on_white: bool = True) -> torch.Tensor:
    """Map per-Gaussian 2D screen flow (fx, fy) to RGB via the colour wheel.

    on_white=True  -> Middlebury look: zero flow is white, fast flow saturated.
    on_white=False -> glow look: zero flow is black (for additive blending).
    """
    wheel = _color_wheel()
    ncols = wheel.shape[0]
    mag = torch.sqrt(fx * fx + fy * fy)
    ang = torch.atan2(-fy, -fx) / math.pi                       # [-1, 1]
    fk = (ang + 1.0) / 2.0 * (ncols - 1)                        # [0, ncols-1]
    k0 = torch.floor(fk).long().clamp(0, ncols - 1)
    k1 = (k0 + 1) % ncols
    f = (fk - k0.float()).unsqueeze(1)
    col = (1.0 - f) * wheel[k0] + f * wheel[k1]                 # (M, 3)
    rad = (mag / max_mag).clamp(0.0, 1.0).unsqueeze(1)
    if on_white:
        col = 1.0 - rad * (1.0 - col)                          # white at rad=0
    else:
        col = rad * col                                        # black at rad=0
    return col.clamp(0.0, 1.0)


def _project(xyz: torch.Tensor, K: torch.Tensor,
             R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Project world-space points (M, 3) to pixel coordinates (M, 2)."""
    cam = xyz @ R.mT + T.reshape(1, 3)                          # (M, 3)
    z = cam[:, 2:3].clamp_min(1e-6)
    uvw = (cam / z) @ K.mT                                      # (M, 3)
    return uvw[:, :2]


def render_rgb_array(model, batch) -> torch.Tensor:
    """Ordinary colour render -> (H, W, 3) float tensor in [0, 1]."""
    output = model(batch)
    rgb = output.rgb_map                                       # (B, H*W, 3)
    H, W = int(batch.meta.H[0].item()), int(batch.meta.W[0].item())
    return rgb[0].reshape(H, W, 3).clamp(0, 1)


def render_flow_array(model, batch, on_white: bool = True) -> torch.Tensor:
    """Render the per-Gaussian motion field -> (H, W, 3) float tensor in [0, 1].

    Each 4D Gaussian carries an implicit linear velocity v = Sigma_xt / Sigma_tt
    (the space-time cross-covariance). We advance every active Gaussian by one
    inter-frame interval, measure its on-screen displacement, colour-code that
    2D flow with the Middlebury wheel, and rasterise through the same splatter.
    """
    sampler = model.sampler
    g = sampler.gaussians
    t = float(batch.t.reshape(-1)[0].item())

    # Eq. 7: indices of Gaussians active at this timestamp (active-subset path)
    idx = sampler.hierarchy.active_indices(t)

    # Sec. 3.2.1: condition only the active 4D Gaussians into 3D at time t
    xyz3, cov6, occ1 = g.get_conditional_3d(t, idx)

    # implicit per-Gaussian velocity v = Sigma_xt / Sigma_tt (active subset)
    cov4 = g.get_cov4(idx)                                     # (M, 4, 4)
    vel = (cov4[:, :3, 3:4] / cov4[:, 3:4, 3:4].clamp_min(1e-12))[..., 0]  # (M, 3)

    # screen-space flow over one inter-frame step
    K, R, T = batch.K[0], batch.R[0], batch.T[0]
    dt = 1.0 / max(1, STATE.frames - 1)
    flow = _project(xyz3 + vel * dt, K, R, T) - _project(xyz3, K, R, T)  # (M, 2)
    mag = flow.norm(dim=1)
    if mag.numel():
        sample = mag
        if sample.numel() > (1 << 24):  # torch.quantile input size limit
            sel = torch.randint(0, sample.numel(), (1 << 20,), device=sample.device)
            sample = sample[sel]
        max_mag = torch.quantile(sample, 0.95).clamp_min(1.0)
    else:
        max_mag = torch.ones((), device=xyz3.device)
    color = flow_uv_to_rgb(flow[:, 0], flow[:, 1], max_mag, on_white=on_white)

    camera = to_x(prepare_gaussian_camera(batch), torch.float)
    rgb, acc, _, _ = render_diff_gauss(xyz3, color, cov6, occ1, camera)
    img = rgb[0]                                               # (H, W, 3)
    if on_white:
        img = img + (1.0 - acc[0])                             # white background
    return img.clamp(0, 1)


def render_to_jpeg(model, batch, mode: str = "rgb", quality: int = 80) -> bytes:
    # VolumetricVideoModel.render_rays does `del batch.output` after copying the
    # reference into `output`, so colour data is read from the return value.
    with torch.inference_mode():
        if mode == "flow":
            img = render_flow_array(model, batch, on_white=True)
        elif mode == "blend":
            scene = render_rgb_array(model, batch)
            grey = scene.mean(dim=-1, keepdim=True).expand_as(scene) * 0.55
            glow = render_flow_array(model, batch, on_white=False)
            img = (grey + glow).clamp(0, 1)
        else:
            img = render_rgb_array(model, batch)
    img = (img.cpu().numpy() * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


async def send_info(websocket):
    """Push the current model + checkpoint list to a client (text frame)."""
    await websocket.send(json.dumps({
        "type": "info",
        "checkpoints": list_checkpoints(),
        "ckpt": STATE.ckpt,
        "frames": STATE.frames,
        "fps": STATE.fps,
        "gaussians": STATE.gaussians,
    }))


async def handle_command(websocket, req: dict):
    """Handle a non-render control message ({'cmd': ...})."""
    cmd = req.get("cmd")
    if cmd in ("info", "list", "hello"):
        await send_info(websocket)
    elif cmd == "load":
        log(f"client requested checkpoint: {req.get('ckpt')}")
        # Run the (multi-second) load in a worker thread so the event loop
        # keeps answering keepalive pings for other clients.
        async with GPU_LOCK:
            await asyncio.to_thread(load_checkpoint, req["ckpt"], req.get("frames"))
        await send_info(websocket)
    elif cmd == "set_frames":
        set_sequence_frames(req.get("frames", STATE.frames))
        await send_info(websocket)
    else:
        await websocket.send(json.dumps({"error": f"unknown command: {cmd}"}))


async def handle_client(websocket):
    log(f"client connected: {websocket.remote_address}")
    await send_info(websocket)
    rendered = 0
    t0 = time.perf_counter()
    try:
        async for message in websocket:
            try:
                req = json.loads(message)
                if isinstance(req, dict) and "cmd" in req:
                    await handle_command(websocket, req)
                    continue
                batch = build_render_batch(req)
                mode = str(req.get("mode", "rgb"))
                async with GPU_LOCK:
                    jpeg = await asyncio.to_thread(render_to_jpeg, MODEL, batch, mode)
                await websocket.send(jpeg)
                rendered += 1
                if time.perf_counter() - t0 > 5:
                    fps = rendered / (time.perf_counter() - t0)
                    log(f"server: {fps:.1f} fps")
                    rendered = 0
                    t0 = time.perf_counter()
            except Exception as e:
                import traceback
                traceback.print_exc()
                await websocket.send(json.dumps({"error": str(e)}))
    finally:
        log(f"client disconnected: {websocket.remote_address}")


async def main():
    global MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; 0.0.0.0 exposes the server to the "
                         "network (unsafe: 'load' unpickles checkpoints)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt",
                    default="gaussianth_flame_salmon_1200/2026-05-21_0141_1200f_latest.pt",
                    help="initial checkpoint (abs path or relative to CKPT_ROOT)")
    ap.add_argument("--frames", type=int, default=0,
                    help="initial sequence length (0 = auto-guess from name)")
    ap.add_argument("--fps", type=int, default=30,
                    help="sequence playback fps reported to the viewer")
    # Parse the real command line saved before EasyVolcap argv injection.
    args, _ = ap.parse_known_args(_CLI_ARGV)

    STATE.fps = args.fps
    MODEL = build_model()
    load_checkpoint(args.ckpt, args.frames or None, trusted=True)
    log(f"starting WebSocket server on ws://{args.host}:{args.port}")

    async with websockets.serve(handle_client, args.host, args.port,
                                max_size=None, ping_interval=20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
