"""
4D Gaussian primitive for the Temporal Gaussian Hierarchy reproduction.

Implements the 4D Gaussian representation of Yang et al. 2023b ("Real-time
Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian
Splatting"), as used by Xu et al., SIGGRAPH Asia 2024 ("Representing Long
Volumetric Video with Temporal Gaussian Hierarchy"), Sec. 3.2.1.

A 4D Gaussian stores a 4D mean (xyz + t), a 4D scale, a scalar opacity, two
isotropic quaternions (q_l, q_r) parameterising a 4D rotation, a base color and
residual SH coefficients. Rendering at a timestamp t is done by conditioning the
4D Gaussian into a 3D Gaussian (standard multivariate-normal conditioning) and
handing the result to a 3D Gaussian rasterizer.
"""

import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.utils.console_utils import log
from easyvolcap.utils.net_utils import make_buffer, make_params
from easyvolcap.utils.gaussian_utils import strip_symmetric, inverse_sigmoid, rgb2sh0


def build_quat_left_matrix(q: torch.Tensor) -> torch.Tensor:
    """Left quaternion-multiplication matrix L(q): (q * v) == L(q) @ v. q: (..., 4) wxyz."""
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    L = torch.stack([
        a, -b, -c, -d,
        b,  a, -d,  c,
        c,  d,  a, -b,
        d, -c,  b,  a,
    ], dim=-1)
    return L.reshape(*q.shape[:-1], 4, 4)


def build_quat_right_matrix(q: torch.Tensor) -> torch.Tensor:
    """Right quaternion-multiplication matrix R(q): (v * q) == R(q) @ v. q: (..., 4) wxyz."""
    p, q1, r, s = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = torch.stack([
        p, -q1, -r, -s,
        q1,  p,  s, -r,
        r,  -s,  p,  q1,
        s,   r, -q1,  p,
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 4, 4)


def build_rotation_4d(q_l: torch.Tensor, q_r: torch.Tensor) -> torch.Tensor:
    """4D rotation matrix from a pair of unit quaternions: M = L(q_l) @ R(q_r). (..., 4, 4)."""
    q_l = F.normalize(q_l, dim=-1)
    q_r = F.normalize(q_r, dim=-1)
    return build_quat_left_matrix(q_l) @ build_quat_right_matrix(q_r)


class GaussianModel4D(nn.Module):
    """A set of 4D Gaussians. Stored parameters use the `_` prefix; public
    `get_*` properties apply the activation functions."""

    def __init__(self,
                 xyz: torch.Tensor = None,            # (N, 3) spatial means
                 colors: torch.Tensor = None,         # (N, 3) initial RGB
                 times: torch.Tensor = None,          # (N, 1) temporal means, normalized [0, 1]
                 init_occ: float = 0.1,
                 init_scale: torch.Tensor = None,     # (N, 3) spatial scale (std, world units)
                 init_scale_t: float = 0.1,           # temporal scale (std, normalized time units)
                 sh_deg: int = 3,
                 ):
        super().__init__()
        self.max_sh_degree = sh_deg
        self.active_sh_degree = make_buffer(torch.zeros(1))
        self.create_from_pcd(xyz, colors, times, init_occ, init_scale, init_scale_t)
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    # ------------------------------------------------------------------ setup
    def _reset_stat_buffers(self, N: int, device):
        self.xyz_gradient_accum = make_buffer(torch.zeros(N, 1, device=device))
        self.denom = make_buffer(torch.zeros(N, 1, device=device))
        self.max_radii2D = make_buffer(torch.zeros(N, device=device))

    def create_from_pcd(self, xyz, colors, times, init_occ, init_scale, init_scale_t):
        if xyz is None:
            xyz = torch.empty(1, 3, device='cuda')
        N = xyz.shape[0]
        device = xyz.device

        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2), device=device)
        if colors is not None:
            features[:, :3, 0] = rgb2sh0(colors)

        if times is None:
            times = torch.full((N, 1), 0.5, device=device)
        elif times.ndim == 1:
            times = times[:, None]

        if init_scale is None:
            init_scale = torch.full((N, 3), 1e-2, device=device)
        scaling = torch.log(init_scale.clamp_min(1e-8))                       # exp activation
        scaling_t = torch.log(torch.full((N, 1), float(init_scale_t), device=device))

        rot_l = torch.zeros((N, 4), device=device); rot_l[:, 0] = 1.0
        rot_r = torch.zeros((N, 4), device=device); rot_r[:, 0] = 1.0

        if not isinstance(init_occ, torch.Tensor):
            init_occ = torch.full((N, 1), float(init_occ), device=device)
        opacity = inverse_sigmoid(init_occ.clamp(1e-4, 1 - 1e-4))

        self._xyz = make_params(xyz)
        self._t = make_params(times)
        self._scaling = make_params(scaling)
        self._scaling_t = make_params(scaling_t)
        self._rotation_l = make_params(rot_l)
        self._rotation_r = make_params(rot_r)
        self._opacity = make_params(opacity)
        self._features_dc = make_params(features[:, :, :1].transpose(1, 2).contiguous())     # (N,1,3)
        self._features_rest = make_params(features[:, :, 1:].transpose(1, 2).contiguous())   # (N,rest,3)

        # densification statistics (one row per Gaussian)
        self._reset_stat_buffers(N, device)

    @torch.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
        # Allow loading checkpoints with a different number of Gaussians.
        if prefix != '' and not prefix.endswith('.'):
            prefix = prefix + '.'
        for name, params in self.named_parameters():
            key = f'{prefix}{name}'
            if key in state_dict:
                params.data = params.data.new_empty(state_dict[key].shape)

    # ------------------------------------------------------------- properties
    @property
    def device(self): return self._xyz.device

    @property
    def get_xyz(self): return self._xyz

    @property
    def get_t(self): return self._t

    @property
    def get_scaling(self): return torch.exp(self._scaling)

    @property
    def get_scaling_t(self): return torch.exp(self._scaling_t)

    @property
    def get_rotation_l(self): return F.normalize(self._rotation_l, dim=-1)

    @property
    def get_rotation_r(self): return F.normalize(self._rotation_r, dim=-1)

    @property
    def get_opacity(self): return torch.sigmoid(self._opacity)

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)   # (N, (deg+1)^2, 3)

    def oneup_sh_degree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def __len__(self):
        return self._xyz.shape[0]

    # --------------------------------------------- Compact Appearance Model
    # Section 3.3 of the paper. Each Gaussian's residual SH coefficients h are
    # zero-initialised; a backward hook on `_features_rest` applies Eq. 11:
    #     g'_h = g_h   if  ||g_h||_2 >= g_th  OR  ||h||_2 != 0
    #            0     otherwise
    # so only Gaussians whose gradient consistently exceeds `g_th` (or which
    # have already been activated) update their SH. Once the proportion of
    # activated Gaussians reaches `lambda_h`, `g_th` is set to infinity to
    # prevent any further activation.
    def enable_compact_appearance(self, gradient_threshold: float = 1e-6):
        # Kept as plain Python attributes so they do not enter state_dict and
        # remain compatible with checkpoints saved before the compact-appearance
        # mechanism existed.
        self.gradient_threshold = float(gradient_threshold)
        self.compact_appearance_enabled = True

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if not self.compact_appearance_enabled:
                return grad
            with torch.no_grad():
                sh_norm = self._features_rest.flatten(1).norm(dim=1)          # (N,)
                if self.gradient_threshold == float('inf'):
                    keep = sh_norm != 0
                else:
                    g_norm = grad.flatten(1).norm(dim=1)                      # (N,)
                    keep = (g_norm >= self.gradient_threshold) | (sh_norm != 0)
            return grad * keep.view(-1, 1, 1).to(grad.dtype)

        # remove any previous hook so this is idempotent
        if hasattr(self, '_compact_handle') and self._compact_handle is not None:
            self._compact_handle.remove()
        self._compact_handle = self._features_rest.register_hook(hook)

    @torch.no_grad()
    def compact_appearance_proportion(self) -> float:
        """Fraction of Gaussians whose residual SH has been activated."""
        sh_norm = self._features_rest.flatten(1).norm(dim=1)
        return (sh_norm != 0).float().mean().item()

    @torch.no_grad()
    def update_compact_appearance(self, lambda_h: float) -> float:
        """Freeze further SH activation once the proportion reaches `lambda_h`.
        Returns the current activated proportion."""
        prop = self.compact_appearance_proportion()
        if prop >= lambda_h:
            self.gradient_threshold = float('inf')
        return prop

    # =========================================================== densification
    # The 4D analogue of 3DGS adaptive control (clone / split / prune). The
    # pattern follows EasyVolcap's GaussianModel: `param.set_(new_storage)`
    # mutates parameter storage in place (Python object identity preserved)
    # so the optimizer keeps tracking the same param object; an
    # `optimizer_state` dict propagated by the sampler is patched in parallel
    # and applied to Adam via `update_optimizer_state`.

    PARAM_NAMES = ('_xyz', '_t', '_scaling', '_scaling_t',
                   '_rotation_l', '_rotation_r', '_opacity',
                   '_features_dc', '_features_rest')

    @torch.no_grad()
    def add_densification_stats(self, viewspace_point_tensor: torch.Tensor,
                                update_filter: torch.Tensor,
                                indices: torch.Tensor):
        """Accumulate per-Gaussian view-space gradient magnitude.

        `viewspace_point_tensor` has gradients only for the active subset
        (size M = active count). `indices` maps the active rows back to the
        global Gaussian indices (size M)."""
        if viewspace_point_tensor.grad is None:
            return
        grad = viewspace_point_tensor.grad[update_filter, :2]               # (m, 2)
        norms = grad.norm(dim=-1, keepdim=True)                              # (m, 1)
        global_idx = indices[update_filter]                                  # (m,)
        self.xyz_gradient_accum[global_idx] += norms
        self.denom[global_idx] += 1

    @torch.no_grad()
    def _drop_rows(self, mask: torch.Tensor, optimizer_state: dict,
                   orig_idx: torch.Tensor) -> torch.Tensor:
        """Remove rows where `mask` is True. Maintains the `orig_idx` mapping
        (current row -> original index, -1 if newly added) and updates the
        running `optimizer_state` masks consistently so that
        `old_keep.sum() == new_keep.sum()` is preserved.
        """
        keep = ~mask
        # mark originals being removed in old_keep
        removed_orig = orig_idx[mask]
        removed_orig = removed_orig[removed_orig >= 0]

        for name in self.PARAM_NAMES:
            p = getattr(self, name)
            p.set_(p.data[keep].detach())
            p.grad = None
        self.xyz_gradient_accum.set_(self.xyz_gradient_accum[keep])
        self.denom.set_(self.denom[keep])
        self.max_radii2D.set_(self.max_radii2D[keep])

        for val in optimizer_state.values():
            val.new_keep = val.new_keep[keep]
            val.new_params = getattr(self, val.name)
            if removed_orig.numel() > 0:
                val.old_keep[removed_orig] = False
        return orig_idx[keep]

    @torch.no_grad()
    def _append_rows(self, new_dict: dict, optimizer_state: dict,
                     orig_idx: torch.Tensor) -> torch.Tensor:
        """Append rows. New rows get `orig_idx = -1` (not original)."""
        n_new = next(iter(new_dict.values())).shape[0]
        for name in self.PARAM_NAMES:
            p = getattr(self, name)
            p.set_(torch.cat([p.data, new_dict[name]], dim=0).detach())
            p.grad = None

        device = self._xyz.device
        N = self._xyz.shape[0]
        self.xyz_gradient_accum.set_(torch.zeros(N, 1, device=device))
        self.denom.set_(torch.zeros(N, 1, device=device))
        self.max_radii2D.set_(torch.zeros(N, device=device))

        for val in optimizer_state.values():
            zeros = torch.zeros(n_new, dtype=torch.bool, device=val.new_keep.device)
            val.new_keep = torch.cat([val.new_keep, zeros], dim=0)
            val.new_params = getattr(self, val.name)

        new_idx = torch.full((n_new,), -1, device=orig_idx.device, dtype=orig_idx.dtype)
        return torch.cat([orig_idx, new_idx])

    @torch.no_grad()
    def densify_and_prune(self, max_grad: float, min_opacity: float,
                          scene_extent: float, percent_dense: float,
                          max_screen_size: float, optimizer_state: dict):
        """Single-call adaptive control: clone (small + high-grad), split
        (large + high-grad), then prune (low opacity / oversize)."""
        device = self._xyz.device
        N0 = self._xyz.shape[0]
        orig_idx = torch.arange(N0, device=device)

        grads = (self.xyz_gradient_accum / self.denom.clamp_min(1)).squeeze(-1)  # (N0,)
        grads[grads.isnan()] = 0.0

        # --- CLONE: small scale, high gradient -> duplicate ----------------
        scaling_max = self.get_scaling.max(dim=1).values                          # (N0,)
        sel_clone = (grads >= max_grad) & (scaling_max <= percent_dense * scene_extent)
        n_clone = int(sel_clone.sum())
        if n_clone > 0:
            new_clone = {n: getattr(self, n)[sel_clone].clone() for n in self.PARAM_NAMES}
            orig_idx = self._append_rows(new_clone, optimizer_state, orig_idx)

        # --- SPLIT: large scale, high gradient -> N children, prune originals ---
        # grads has length N0; pad to the current tensor size (clone appended zero-grad rows)
        N_cur = self._xyz.shape[0]
        padded = torch.zeros(N_cur, device=device)
        padded[:N0] = grads
        cur_scaling_max = self.get_scaling.max(dim=1).values                       # (N_cur,)
        sel_split = (padded >= max_grad) & (cur_scaling_max > percent_dense * scene_extent)
        n_split = int(sel_split.sum())
        if n_split > 0:
            N_split_factor = 2
            s_xyz = self.get_scaling[sel_split].repeat(N_split_factor, 1)
            s_t = self.get_scaling_t[sel_split].repeat(N_split_factor, 1)
            s4 = torch.cat([s_xyz, s_t], dim=-1)
            samples = torch.randn_like(s4) * s4
            R4 = build_rotation_4d(self._rotation_l[sel_split],
                                   self._rotation_r[sel_split]).repeat(N_split_factor, 1, 1)
            offsets = torch.bmm(R4, samples.unsqueeze(-1)).squeeze(-1)
            shrink = 0.8 * N_split_factor       # 1.6, matches 3DGS convention
            new_split = {
                '_xyz': self._xyz[sel_split].repeat(N_split_factor, 1) + offsets[:, :3],
                '_t':   self._t[sel_split].repeat(N_split_factor, 1) + offsets[:, 3:4],
                '_scaling':   torch.log(self.get_scaling[sel_split].repeat(N_split_factor, 1) / shrink),
                '_scaling_t': torch.log(self.get_scaling_t[sel_split].repeat(N_split_factor, 1) / shrink),
                '_rotation_l': self._rotation_l[sel_split].repeat(N_split_factor, 1),
                '_rotation_r': self._rotation_r[sel_split].repeat(N_split_factor, 1),
                '_opacity':    self._opacity[sel_split].repeat(N_split_factor, 1),
                '_features_dc':   self._features_dc[sel_split].repeat(N_split_factor, 1, 1),
                '_features_rest': self._features_rest[sel_split].repeat(N_split_factor, 1, 1),
            }
            orig_idx = self._append_rows(new_split, optimizer_state, orig_idx)

            # prune the original selected (still at their old positions; new rows were appended)
            prune_split = torch.zeros(self._xyz.shape[0], dtype=torch.bool, device=device)
            prune_split[:N_cur] = sel_split
            orig_idx = self._drop_rows(prune_split, optimizer_state, orig_idx)

        # --- PRUNE: low opacity / oversize --------------------------------
        prune_mask = (self.get_opacity < min_opacity).squeeze(-1)
        if max_screen_size is not None and max_screen_size > 0:
            big_ws = self.get_scaling.max(dim=1).values > 0.1 * scene_extent
            big_vs = self.max_radii2D > max_screen_size
            prune_mask = prune_mask | big_ws | big_vs
        n_pruned = int(prune_mask.sum())
        if n_pruned > 0:
            orig_idx = self._drop_rows(prune_mask, optimizer_state, orig_idx)

        # sanity check
        for val in optimizer_state.values():
            assert val.old_keep.sum() == val.new_keep.sum(), \
                f'{val.name}: old_keep.sum={val.old_keep.sum().item()} != new_keep.sum={val.new_keep.sum().item()}'

        torch.cuda.empty_cache()
        return n_clone, n_split, n_pruned

    # ----------------------------------------------------------- 4D covariance
    def get_cov4(self) -> torch.Tensor:
        """Full 4D covariance Sigma = R S S^T R^T. Returns (N, 4, 4)."""
        s = torch.cat([self.get_scaling, self.get_scaling_t], dim=-1)        # (N, 4) std
        R = build_rotation_4d(self._rotation_l, self._rotation_r)            # (N, 4, 4)
        S2 = torch.diag_embed(s * s)                                        # (N, 4, 4)
        return R @ S2 @ R.transpose(-1, -2)

    def get_marginal_t(self):
        """Temporal marginal: mean mu_t (N,1) and variance var_t (N,1) = Sigma_44."""
        cov4 = self.get_cov4()
        var_t = cov4[:, 3, 3:4]                                             # (N, 1)
        return self._t, var_t

    @torch.no_grad()
    def get_influence_range(self, o_th: float = 0.05):
        """Temporal influence range [tau_lo, tau_hi] (Eq. 4 of the TGH paper).

        r = sqrt(-2 * ln(o_th) * sigma_t), with sigma_t = Sigma_44 the temporal
        variance. Returns (tau_lo (N,), tau_hi (N,)) and radius r (N,)."""
        mu_t, var_t = self.get_marginal_t()
        mu_t = mu_t[:, 0]
        var_t = var_t[:, 0].clamp_min(1e-12)
        r = torch.sqrt(-2.0 * torch.log(torch.tensor(o_th, device=var_t.device)) * var_t)
        return mu_t - r, mu_t + r, r

    # ------------------------------------------------- conditional 3D Gaussian
    def get_conditional_3d(self, timestamp: float):
        """Condition the 4D Gaussians at `timestamp` into 3D Gaussians.

        Standard multivariate-normal conditioning with the partition
            Sigma = [[Sigma_xx (3x3), Sigma_xt (3x1)],
                     [Sigma_tx (1x3), Sigma_tt (1x1)]].

        Returns:
            xyz3 (N, 3)  - conditional 3D means
            cov6 (N, 6)  - upper-triangular 3D covariance (diff_gauss layout)
            occ1 (N, 1)  - opacity scaled by the temporal marginal at `timestamp`
        """
        cov4 = self.get_cov4()                                              # (N, 4, 4)
        Sigma_xx = cov4[:, :3, :3]                                          # (N, 3, 3)
        Sigma_xt = cov4[:, :3, 3:4]                                         # (N, 3, 1)
        Sigma_tx = cov4[:, 3:4, :3]                                         # (N, 1, 3)
        Sigma_tt = cov4[:, 3:4, 3:4].clamp_min(1e-12)                       # (N, 1, 1)
        inv_tt = 1.0 / Sigma_tt                                             # (N, 1, 1)

        dt = timestamp - self._t                                           # (N, 1)

        # conditional mean: mu_xyz + Sigma_xt Sigma_tt^-1 (t - mu_t)
        shift = (Sigma_xt @ inv_tt)[..., 0] * dt                            # (N, 3)
        xyz3 = self._xyz + shift

        # conditional covariance: Sigma_xx - Sigma_xt Sigma_tt^-1 Sigma_tx
        cov3 = Sigma_xx - Sigma_xt @ inv_tt @ Sigma_tx                      # (N, 3, 3)
        cov6 = strip_symmetric(cov3)                                        # (N, 6)

        # temporal marginal opacity factor: exp(-0.5 (t - mu_t)^2 / Sigma_tt)
        var_t = Sigma_tt[:, :, 0]                                           # (N, 1)
        opacity_factor = torch.exp(-0.5 * dt * dt / var_t)                  # (N, 1)
        occ1 = self.get_opacity * opacity_factor                            # (N, 1)

        return xyz3, cov6, occ1


if __name__ == '__main__':
    # Quick self-test of the 4D Gaussian math.
    torch.manual_seed(0)
    N = 1000
    xyz = torch.randn(N, 3, device='cuda')
    colors = torch.rand(N, 3, device='cuda')
    times = torch.rand(N, 1, device='cuda')
    gm = GaussianModel4D(xyz, colors, times, init_occ=0.5,
                         init_scale=torch.full((N, 3), 0.05, device='cuda'),
                         init_scale_t=0.1, sh_deg=3).cuda()

    cov4 = gm.get_cov4()
    assert cov4.shape == (N, 4, 4)
    # covariance must be symmetric PSD
    assert torch.allclose(cov4, cov4.transpose(-1, -2), atol=1e-5), 'cov4 not symmetric'
    eig = torch.linalg.eigvalsh(cov4)
    assert (eig > -1e-5).all(), f'cov4 not PSD, min eig {eig.min().item()}'

    xyz3, cov6, occ1 = gm.get_conditional_3d(0.5)
    assert xyz3.shape == (N, 3) and cov6.shape == (N, 6) and occ1.shape == (N, 1)

    # opacity factor must peak when timestamp == mu_t
    _, _, occ_at_mu = [], [], None
    occ_far = gm.get_conditional_3d(times.mean().item() + 5.0)[2]
    occ_near = gm.get_conditional_3d(times.mean().item())[2]
    assert occ_near.mean() > occ_far.mean(), 'opacity should decay away from mu_t'

    lo, hi, r = gm.get_influence_range(0.05)
    assert (hi >= lo).all() and (r >= 0).all()

    log('GaussianModel4D self-test passed: '
        f'cov4 PSD, conditional-3D shapes OK, opacity decay OK, influence range OK')

    # Compact Appearance Model: gradient-thresholding hook (Eq. 11).
    gm2 = GaussianModel4D(xyz, colors, times, init_occ=0.5,
                          init_scale=torch.full((N, 3), 0.05, device='cuda'),
                          init_scale_t=0.1, sh_deg=3).cuda()
    # gradient of 1.0 per element gives per-row L2 norm sqrt(rest*3); pick a
    # threshold well above that so the hook zeros every diffuse row.
    rest = gm2._features_rest.shape[1]
    huge_th = (rest * 3) ** 0.5 + 1.0
    gm2.enable_compact_appearance(gradient_threshold=huge_th)

    out = (gm2._features_rest * 1.0).sum()
    out.backward()
    g = gm2._features_rest.grad.flatten(1).norm(dim=1)
    assert (g == 0).all(), f'expected all gradients zeroed, got nonzero count {(g!=0).sum().item()}'
    log('Compact-appearance gradient hook zeros sub-threshold updates OK')

    # activate a few Gaussians (give them nonzero SH), then verify those still get gradient through
    with torch.no_grad():
        gm2._features_rest[:5].fill_(0.1)
    gm2._features_rest.grad = None
    out = (gm2._features_rest * 1.0).sum()
    out.backward()
    g = gm2._features_rest.grad.flatten(1).norm(dim=1)
    assert (g[:5] > 0).all(), 'already-activated rows must keep their gradient'
    assert (g[5:] == 0).all(), 'still-diffuse rows must remain zero'

    # lambda_h cutoff
    prop = gm2.update_compact_appearance(lambda_h=0.001)         # 5/1000 == 0.005 > 0.001
    assert gm2.gradient_threshold == float('inf'), \
        f'gradient_threshold should be inf after cutoff, got {gm2.gradient_threshold}'
    log(f'lambda_h cutoff: proportion {prop:.3f}, gradient_threshold -> inf OK')
