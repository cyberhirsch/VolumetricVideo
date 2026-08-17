#!/usr/bin/env python
"""Generate a numerical parity fixture for VoluLab's TGH port.

Builds a small random GaussianModel4D + TemporalGaussianHierarchy with the
real training classes, then records what they say: the active row set and
the conditional 3D gaussians (mean, covariance, alpha) at a spread of query
times, including exact segment boundaries where float handling matters.

Writes two files next to each other:
  fixture.npz   the model as a checkpoint-shaped npz (production key names),
                loadable by VoluLab through its normal .npz import
  fixture.json  the expected outputs, for the browser to compare against

Run inside the training environment (WSL, conda env lvv):
  python scripts/gen_parity_fixture.py --easyvolcap "/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap" --out /tmp/tgh-fixture

CPU only - no CUDA, no rasterizer.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--easyvolcap', default='/mnt/g/3D Generation/Long Volumetric Video/EasyVolcap')
    ap.add_argument('--out', default='/tmp/tgh-fixture')
    ap.add_argument('--count', type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, args.easyvolcap)
    from easyvolcap.utils.gaussian4d_utils import GaussianModel4D
    from easyvolcap.utils.temporal_gaussian_hierarchy import TemporalGaussianHierarchy

    torch.manual_seed(7)
    n = args.count

    model = GaussianModel4D(
        xyz=torch.randn(n, 3),
        colors=torch.rand(n, 3),
        times=torch.rand(n, 1),
        init_occ=0.5,
        init_scale=torch.rand(n, 3) * 0.2 + 0.01,
        init_scale_t=0.1,
        sh_deg=3,
    )

    # non-identity everything: random rotation pairs, spread temporal scales,
    # non-zero higher SH bands - the port has to survive all of it
    with torch.no_grad():
        model._rotation_l.copy_(torch.randn(n, 4))
        model._rotation_r.copy_(torch.randn(n, 4))
        model._scaling_t.copy_(torch.log(torch.rand(n, 1) * 0.3 + 0.005))
        model._opacity.copy_(torch.randn(n, 1))
        model._features_dc.copy_(torch.randn(n, 1, 3))
        model._features_rest.copy_(torch.randn(n, 15, 3) * 0.1)

    hierarchy = TemporalGaussianHierarchy(num_levels=4, root_segment=0.5, o_th=0.05)
    hierarchy.assign(model)

    # checkpoint-shaped npz, so VoluLab's production load path reads it
    arrays = {}
    for name, p in model.named_parameters():
        arrays[f'sampler.gaussians.{name}'] = p.detach().numpy()
    for name in ('level', 'segment', 'seg_len'):
        arrays[f'sampler.hierarchy.{name}'] = getattr(hierarchy, name).detach().numpy()

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'fixture.npz'), **arrays)

    # expected outputs; boundary times (0.25, 0.5, 1.0) included on purpose
    times = [0.0, 0.13, 0.25, 0.3, 0.5, 0.55, 0.999, 1.0]
    cases = []
    for t in times:
        idx = hierarchy.active_indices(t)
        xyz3, cov6, occ1 = model.get_conditional_3d(t, idx)
        feat = model.get_features_at(idx)                       # (M, 16, 3)
        f_all = feat.transpose(1, 2).flatten(1)                 # channel-major, as save_ply
        cases.append({
            't': t,
            'indices': idx.tolist(),
            'xyz': xyz3.detach().numpy().tolist(),
            'cov6': cov6.detach().numpy().tolist(),
            'alpha': occ1[:, 0].detach().numpy().tolist(),
            'f_dc': f_all[:, 0::16].detach().numpy().tolist(),
            'f_rest': torch.cat([f_all[:, c * 16 + 1:(c + 1) * 16] for c in range(3)], dim=1).detach().numpy().tolist(),
        })

    with open(os.path.join(args.out, 'fixture.json'), 'w') as f:
        json.dump({'count': n, 'cases': cases}, f)

    counts = ', '.join(f'{c["t"]}:{len(c["indices"])}' for c in cases)
    print(f'wrote {args.out}/fixture.npz + fixture.json ({n} gaussians; active per t: {counts})')


if __name__ == '__main__':
    main()
