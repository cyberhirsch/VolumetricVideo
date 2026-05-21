"""
Temporal Gaussian Hierarchy (TGH).

Core data structure of Xu et al., SIGGRAPH Asia 2024, "Representing Long
Volumetric Video with Temporal Gaussian Hierarchy" (Sec. 3.2).

The hierarchy has `L` levels. Level `l` partitions the (normalized) time axis
[0, 1] into equal segments of length `s_l = S / 2**l`, where `S` is the root
segment length. A single global segment of infinite length captures fully
static content.

Each 4D Gaussian is placed in the *shortest* segment that still fully contains
its temporal influence range [tau_lo, tau_hi] (Eq. 6). Because level-`l` segment
boundaries are a subset of level-`(l+1)` boundaries, segments are nested: a
Gaussian that fits inside one level-`l` segment also fits inside its coarser
ancestors, so "shortest containing segment" == "finest level at which both
endpoints fall in the same segment".

At a query time `t`, exactly one segment per level is active, with index
`n_l = floor(t / s_l)` (Eq. 7). A Gaussian is active iff it was assigned to that
level's active segment, or it lives in the global segment.

This class is a lightweight *index* over a `GaussianModel4D`: it does not own the
Gaussian parameters. The owner (the sampler) calls `assign()` after
initialisation and after every densification, then `query()` every iteration.

The placement strategy is pluggable (`segmentation`): "uniform" reproduces the
paper; "adaptive" is reserved for the content-adaptive extension.
"""

import torch
from torch import nn

from easyvolcap.utils.console_utils import log
from easyvolcap.utils.net_utils import make_buffer
from easyvolcap.utils.gaussian4d_utils import GaussianModel4D

GLOBAL_LEVEL = -1   # sentinel level id for the infinite global segment


class TemporalGaussianHierarchy(nn.Module):
    def __init__(self,
                 num_levels: int = 9,        # L
                 root_segment: float = 0.25,  # S, normalized fraction of the [0,1] time axis
                 o_th: float = 0.05,          # opacity threshold for the influence range (Eq. 4)
                 segmentation: str = 'uniform',
                 ):
        super().__init__()
        assert num_levels >= 1
        assert 0.0 < root_segment <= 1.0
        assert segmentation in ('uniform', 'adaptive')
        self.num_levels = num_levels
        self.root_segment = root_segment
        self.o_th = o_th
        self.segmentation = segmentation

        # s_l = S / 2**l, finest level last
        seg_len = torch.tensor([root_segment / (2 ** l) for l in range(num_levels)])
        self.seg_len = make_buffer(seg_len)                  # (L,)

        # per-Gaussian assignment, (re)sized by assign()
        self.level = make_buffer(torch.zeros(0, dtype=torch.long))     # (N,) in [0, L) or GLOBAL_LEVEL
        self.segment = make_buffer(torch.zeros(0, dtype=torch.long))   # (N,) segment index within its level

        # Allow loading a checkpoint where N differs from the current size
        # (after every densification N grows; on load we just reallocate).
        self._register_load_state_dict_pre_hook(self._lsd_pre_hook)

    @torch.no_grad()
    def _lsd_pre_hook(self, state_dict, prefix, *_):
        for name in ('level', 'segment'):
            k = f'{prefix}{name}'
            if k in state_dict:
                buf = getattr(self, name)
                if buf.shape != state_dict[k].shape:
                    buf.data = buf.data.new_empty(state_dict[k].shape)

    # --------------------------------------------------------------- placement
    @torch.no_grad()
    def assign(self, model: GaussianModel4D):
        """Place every Gaussian of `model` in the hierarchy (Eq. 6).

        A Gaussian fits in a single level-`l` segment iff both endpoints of its
        influence range fall in the same segment, i.e.
            floor(tau_lo / s_l) == floor(tau_hi / s_l).
        Its level is the finest `l` for which this holds; if none holds (the
        range straddles every root segment boundary or is wider than S) it goes
        to the global segment.
        """
        tau_lo, tau_hi, _ = model.get_influence_range(self.o_th)         # (N,), (N,)
        # clamp into [0, 1]; content outside the captured window is still placed
        tau_lo = tau_lo.clamp(0.0, 1.0)
        tau_hi = tau_hi.clamp(0.0, 1.0)
        N = tau_lo.shape[0]
        device = tau_lo.device

        seg_len = self.seg_len.to(device)                                # (L,)
        lo_idx = torch.floor(tau_lo[:, None] / seg_len[None, :]).long()   # (N, L)
        hi_idx = torch.floor(tau_hi[:, None] / seg_len[None, :]).long()   # (N, L)
        fits = lo_idx == hi_idx                                          # (N, L) bool

        if self.segmentation == 'adaptive':
            # Reserved: content-adaptive boundaries (docs/limitations-and-extensions.md).
            # Falls back to uniform until implemented.
            pass

        # finest level (largest l) where the Gaussian fits in one segment
        level_ids = torch.arange(self.num_levels, device=device)         # (L,)
        masked = torch.where(fits, level_ids[None, :], torch.full_like(lo_idx, -1))
        best_level = masked.max(dim=1).values                            # (N,) in [-1, L)

        has_home = best_level >= 0
        level = torch.where(has_home, best_level,
                            torch.full_like(best_level, GLOBAL_LEVEL))   # (N,)

        # segment index within the assigned level (0 for global)
        safe_level = best_level.clamp_min(0)
        seg_at_level = lo_idx.gather(1, safe_level[:, None])[:, 0]        # (N,)
        segment = torch.where(has_home, seg_at_level,
                              torch.zeros_like(seg_at_level))            # (N,)

        self.level = make_buffer(level)
        self.segment = make_buffer(segment)
        return level, segment

    # ----------------------------------------------------------------- queries
    @torch.no_grad()
    def active_mask(self, t: float) -> torch.Tensor:
        """Boolean mask (N,) of Gaussians active at normalized time `t` (Eq. 7)."""
        device = self.level.device
        seg_len = self.seg_len.to(device)                                # (L,)
        # active segment per level at time t
        active_seg = torch.floor(torch.tensor(t, device=device) / seg_len).long()  # (L,)

        is_global = self.level == GLOBAL_LEVEL
        safe_level = self.level.clamp_min(0)
        wanted = active_seg[safe_level]                                  # (N,)
        in_active_segment = (~is_global) & (self.segment == wanted)
        return in_active_segment | is_global

    @torch.no_grad()
    def active_indices(self, t: float) -> torch.Tensor:
        """Integer indices of Gaussians active at normalized time `t`."""
        return self.active_mask(t).nonzero(as_tuple=True)[0]

    # ------------------------------------------------------------------- stats
    @torch.no_grad()
    def level_counts(self):
        """Number of Gaussians per level; index L is the global segment."""
        counts = torch.zeros(self.num_levels + 1, dtype=torch.long)
        for l in range(self.num_levels):
            counts[l] = (self.level == l).sum()
        counts[self.num_levels] = (self.level == GLOBAL_LEVEL).sum()
        return counts

    def extra_repr(self):
        return (f'num_levels={self.num_levels}, root_segment={self.root_segment}, '
                f'o_th={self.o_th}, segmentation={self.segmentation}')


if __name__ == '__main__':
    # Self-test of placement (Eq. 6) and query (Eq. 7).
    import math
    torch.manual_seed(0)
    device = 'cuda'
    # 4 levels, root segment 0.5 -> s_l = 0.5, 0.25, 0.125, 0.0625.
    # Root boundaries at 0.0, 0.5, 1.0; every s_l divides 0.5 so t=0.5 is a
    # boundary at every level.
    L, S = 4, 0.5
    tgh = TemporalGaussianHierarchy(num_levels=L, root_segment=S, o_th=0.05).to(device)

    # influence radius r = sqrt(-2 ln(o_th) * var_t); with identity rotation
    # (q_l = q_r = identity at init) var_t = scale_t**2, so r = k * scale_t.
    k = math.sqrt(-2.0 * math.log(0.05))

    def make(ts, radii):
        n = len(ts)
        gm = GaussianModel4D(torch.zeros(n, 3, device=device),
                             torch.ones(n, 3, device=device),
                             torch.tensor(ts, device=device)[:, None],
                             init_occ=0.5,
                             init_scale=torch.full((n, 3), 0.01, device=device),
                             init_scale_t=0.1, sh_deg=0).to(device)
        scale_t = torch.tensor([r / k for r in radii], device=device)[:, None]
        with torch.no_grad():
            gm._scaling_t.copy_(torch.log(scale_t))
        return gm

    # G0: tiny extent at t=0.55  -> finest level 3
    # G1: tiny extent at t=0.50  -> straddles a boundary at every level -> global
    # G2: huge extent            -> global
    # G3: extent [0.25,0.35]     -> fits levels 0,1,2 but not 3 -> finest level 2
    gm = make([0.55, 0.50, 0.50, 0.30], [0.01, 0.01, 2.0, 0.05])
    tgh.assign(gm)
    lvl = tgh.level.tolist()
    seg = tgh.segment.tolist()
    log(f'levels={lvl}  segments={seg}')

    assert lvl[0] == L - 1, f'G0 tiny extent should be finest level {L-1}, got {lvl[0]}'
    assert lvl[1] == GLOBAL_LEVEL, f'G1 straddling t=0.5 should be global, got {lvl[1]}'
    assert lvl[2] == GLOBAL_LEVEL, f'G2 huge extent should be global, got {lvl[2]}'
    assert lvl[3] == 2, f'G3 extent [0.25,0.35] should be level 2, got {lvl[3]}'

    # G0 finest segment: floor(0.55 / 0.0625) = 8
    assert seg[0] == 8, f'G0 segment should be 8, got {seg[0]}'

    # query at t=0.55: G0 active, G3 (level 2 seg floor(0.3/0.125)=2) inactive, globals active
    m = tgh.active_mask(0.55)
    assert m[0].item(), 'G0 should be active at t=0.55'
    assert not m[3].item(), 'G3 should be inactive at t=0.55'
    assert m[1].item() and m[2].item(), 'global Gaussians always active'

    # query at t=0.30: G3 active, G0 inactive
    m2 = tgh.active_mask(0.30)
    assert m2[3].item(), 'G3 should be active at t=0.30'
    assert not m2[0].item(), 'G0 should be inactive at t=0.30'

    counts = tgh.level_counts()
    log(f'level_counts (last entry = global) = {counts.tolist()}')
    log('TemporalGaussianHierarchy self-test passed: placement (Eq.6) and query (Eq.7) OK')
