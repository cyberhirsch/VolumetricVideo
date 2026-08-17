#!/usr/bin/env python
"""Convert a trained TGH checkpoint (.pt) into the .npz VoluLab opens.

VoluLab reads the gaussian and hierarchy tensors straight out of the
checkpoint and evaluates the 4D representation itself, so this is a format
conversion, not an export: no per-frame baking, no rasterizer, no CUDA.

Usage:
  python scripts/export_npz.py <checkpoint.pt> <out.npz> [--frames N] [--fps F]

--frames/--fps are stored as `volulab.meta` so VoluLab can set up its
timeline without asking; omit them and VoluLab asks on load instead.
"""

import argparse

import numpy as np
import torch

# per-gaussian densification statistics, useless outside training
STRIP = ('xyz_gradient_accum', 'denom', 'max_radii2D')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', help='trained checkpoint (.pt)')
    ap.add_argument('out', help='output .npz path')
    ap.add_argument('--frames', type=int, default=0, help='frame count of the captured sequence')
    ap.add_argument('--fps', type=float, default=30.0, help='playback frame rate')
    args = ap.parse_args()

    # checkpoints are trusted local artifacts; they contain pickled objects
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state = ckpt.get('network', ckpt.get('model', ckpt))

    arrays = {}
    for k, v in state.items():
        if ('.gaussians.' in k or '.hierarchy.' in k) and not any(k.endswith(s) for s in STRIP):
            arrays[k] = v.detach().cpu().numpy()

    if not arrays:
        raise SystemExit('no gaussian/hierarchy tensors found - is this a TGH checkpoint?')

    if args.frames > 0:
        arrays['volulab.meta'] = np.array([args.frames, args.fps], dtype=np.float32)

    np.savez_compressed(args.out, **arrays)
    total = sum(a.nbytes for a in arrays.values())
    print(f'wrote {args.out}: {len(arrays)} arrays, {total / 1e6:.1f} MB uncompressed')


if __name__ == '__main__':
    main()
