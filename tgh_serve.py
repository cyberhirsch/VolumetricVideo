#!/usr/bin/env python
"""TGH WebSocket render server.

Loads the trained GaussianTHSampler checkpoint and serves rendered frames
over WebSocket to a browser client (see tgh_viewer.html).

Protocol:
  Client -> Server (JSON):  { K, R, T, H, W, t }
    K: 3x3 intrinsics (list of lists)
    R: 3x3 world-to-camera rotation (list of lists)
    T: 3x1 world-to-camera translation (list of lists)
    H, W: image height/width
    t: normalized time in [0, 1]

  Server -> Client (binary):  JPEG bytes of the rendered RGB frame.

Run as:
  bash serve.sh
or directly:
  python tgh_serve.py [--port 8765] [--ckpt /path/to/latest.pt]
"""

import os
import sys
import io
import json
import argparse
import asyncio
import time
from pathlib import Path

# Inject argv to look like an EasyVolcap test invocation BEFORE importing
# anything from easyvolcap — the engine reads sys.argv at import time.
_EVC_DIR = "/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap"
os.chdir(_EVC_DIR)

sys.argv = [
    "tgh_serve.py",
    "-t", "test",
    "-c", "configs/exps/gaussianth/gaussianth_flame_salmon.yaml",
    # full 300-frame timeline accessible at eval-time
    "val_dataloader_cfg.dataset_cfg.frame_sample=[0,300,1]",
    "val_dataloader_cfg.dataset_cfg.ratio=0.5",
    "dataloader_cfg.dataset_cfg.frame_sample=[0,300,1]",
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


def build_model_with_checkpoint(ckpt_path: str):
    """Construct the VolumetricVideoModel and load weights.

    We deliberately skip the dataloader/runner construction: a render server
    does not need a val_dataloader (which would otherwise spend 5+ minutes
    decoding the entire 300-frame training set into memory).
    """
    model = MODELS.build(cfg.model_cfg).cuda()

    log(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    state = ckpt.get("network", ckpt.get("model", ckpt))
    res = model.load_state_dict(state, strict=False, assign=True)
    if res.missing_keys:
        log(f"missing keys ({len(res.missing_keys)}): {res.missing_keys[:5]} ...")
    if res.unexpected_keys:
        log(f"unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:5]} ...")

    model.eval()
    return model


def build_render_batch(req: dict) -> dotdict:
    """Build the batch dict that GaussianTHSampler.forward expects."""
    H, W = int(req["H"]), int(req["W"])
    K = np.asarray(req["K"], dtype=np.float32)
    R = np.asarray(req["R"], dtype=np.float32)
    T = np.asarray(req["T"], dtype=np.float32)
    if T.ndim == 1:
        T = T[:, None]

    t_norm = float(req["t"])
    # Map normalized t to a frame index in [0, 299]; sample_index_time will
    # then recompute t internally during eval mode.
    n_train_frames = 300
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


def render_to_jpeg(model, batch, quality: int = 80) -> bytes:
    with torch.inference_mode():
        # VolumetricVideoModel.render_rays does `del batch.output` after copying
        # the reference into `output`, so we must read rgb_map from the return
        # value, not from batch.
        output = model(batch)
    rgb = output.rgb_map  # (B, H*W, 3)
    H, W = int(batch.meta.H[0].item()), int(batch.meta.W[0].item())
    img = rgb[0].reshape(H, W, 3).clamp(0, 1).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# Globals so the websocket handler can reach the loaded model
MODEL = None


async def handle_client(websocket):
    log(f"client connected: {websocket.remote_address}")
    rendered = 0
    t0 = time.perf_counter()
    try:
        async for message in websocket:
            try:
                req = json.loads(message)
                batch = build_render_batch(req)
                jpeg = render_to_jpeg(MODEL, batch)
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
                await websocket.send(json.dumps({"error": str(e)}).encode())
    finally:
        log(f"client disconnected: {websocket.remote_address}")


async def main():
    global MODEL

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/lvv-data/trained_model/gaussianth_flame_salmon/latest.pt"))
    # ignore unknown args (those were consumed by EasyVolcap's argparse)
    args, _ = ap.parse_known_args()

    MODEL = build_model_with_checkpoint(args.ckpt)
    log(f"model loaded, sampler has {len(MODEL.sampler.gaussians)} 4D Gaussians")
    log(f"starting WebSocket server on ws://0.0.0.0:{args.port}")

    async with websockets.serve(handle_client, "0.0.0.0", args.port,
                                max_size=None, ping_interval=20):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
