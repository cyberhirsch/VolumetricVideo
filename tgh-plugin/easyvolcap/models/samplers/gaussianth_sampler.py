"""
GaussianTHSampler - Temporal Gaussian Hierarchy sampler.

Reproduction of Xu et al., SIGGRAPH Asia 2024, "Representing Long Volumetric
Video with Temporal Gaussian Hierarchy".

Owns a single flat `GaussianModel4D` (all 4D Gaussians across the whole video)
and a `TemporalGaussianHierarchy` index. Each training/rendering iteration:
  1. read the frame's normalized timestamp t,
  2. query the hierarchy for the Gaussians active at t (one segment per level
     plus the global segment, Eq. 7),
  3. condition those 4D Gaussians into 3D Gaussians at t (Sec. 3.2.1),
  4. rasterize with the 3D differentiable Gaussian rasterizer.

Because only the active subset is rasterized, the per-iteration compute is
governed by the hierarchy depth L, not the video length - the constant-cost
property the paper relies on.
"""

import torch
from torch import nn
from torch.optim import Adam

from easyvolcap.engine import cfg, SAMPLERS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.utils.console_utils import log, yellow_slim, dotdict
from easyvolcap.utils.net_utils import normalize, update_optimizer_state
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.data_utils import to_x
from easyvolcap.utils.gaussian_utils import render_diff_gauss, prepare_gaussian_camera
from easyvolcap.utils.gaussian4d_utils import GaussianModel4D
from easyvolcap.utils.temporal_gaussian_hierarchy import TemporalGaussianHierarchy

from easyvolcap.models.cameras.optimizable_camera import OptimizableCamera
from easyvolcap.models.samplers.point_planes_sampler import PointPlanesSampler
from easyvolcap.models.networks.volumetric_video_network import VolumetricVideoNetwork


@SAMPLERS.register_module()
class GaussianTHSampler(PointPlanesSampler):
    def __init__(self,
                 network: VolumetricVideoNetwork = None,  # unused, kept for API compatibility

                 # 4D Gaussian appearance
                 sh_deg: int = 3,

                 # Temporal Gaussian Hierarchy
                 num_levels: int = 9,
                 root_segment: float = 0.25,   # S, normalized fraction of the [0, 1] time axis
                 o_th: float = 0.05,
                 segmentation: str = 'uniform',

                 # initialization
                 n_init_points: int = 131072,
                 init_occ: float = 0.1,
                 init_scale: float = 0.02,
                 init_scale_t: float = 0.1,

                 # Compact Appearance Model (Sec. 3.3)
                 compact_appearance: bool = True,
                 compact_grad_threshold: float = 1e-6,
                 compact_lambda_h: float = 0.15,
                 compact_check_every: int = 100,

                 # Densification (3DGS-style adaptive control adapted to 4D)
                 densify_from_iter: int = 500,
                 densify_until_iter: int = 15000,
                 densification_interval: int = 100,
                 densify_grad_threshold: float = 2e-4,
                 percent_dense: float = 0.01,
                 min_opacity: float = 0.005,
                 max_screen_size: float = 0.0,        # 0 disables screen-space pruning
                 sh_update_iter: int = 1000,

                 **kwargs,
                 ):
        self.kwargs = dotdict(kwargs)
        # we build our own 4D initialization; never load the parent's per-frame
        # point clouds (and keep its discarded dummy clouds tiny)
        kwargs['skip_loading_points'] = True
        kwargs['n_points'] = 1
        call_from_cfg(super().__init__, kwargs, network=network)

        # the parent builds embedder/regressor modules we do not use
        del self.pcd_embedder
        del self.xyz_embedder
        del self.resd_regressor
        del self.geo_regressor
        del self.dir_embedder
        del self.rgb_regressor
        # the parent's per-frame point clouds are replaced by a single 4D model
        self.pcds = nn.ParameterList()

        self.sh_deg = sh_deg
        self.last_output = None

        # initialize 4D Gaussians: positions uniform in scene bounds, temporal
        # means uniform over the normalized timeline
        bounds = OptimizableCamera.bounds.float()                    # (2, 3)
        lo, hi = bounds[0], bounds[1]
        xyz = torch.rand(n_init_points, 3) * (hi - lo) + lo
        times = torch.rand(n_init_points, 1)
        colors = torch.rand(n_init_points, 3)
        scales = torch.full((n_init_points, 3), float(init_scale))

        self.gaussians = GaussianModel4D(
            xyz.cuda(), colors.cuda(), times.cuda(),
            init_occ=init_occ,
            init_scale=scales.cuda(),
            init_scale_t=init_scale_t,
            sh_deg=sh_deg,
        )

        self.hierarchy = TemporalGaussianHierarchy(
            num_levels=num_levels,
            root_segment=root_segment,
            o_th=o_th,
            segmentation=segmentation,
        )
        self.hierarchy.assign(self.gaussians)
        counts = self.hierarchy.level_counts()
        log(f'TemporalGaussianHierarchy initialized: {n_init_points} 4D Gaussians, '
            f'level counts (last=global) = {counts.tolist()}')

        # Compact Appearance Model
        self.compact_appearance = compact_appearance
        self.compact_lambda_h = compact_lambda_h
        self.compact_check_every = compact_check_every
        if compact_appearance:
            self.gaussians.enable_compact_appearance(compact_grad_threshold)
            log(f'Compact Appearance enabled: g_th={compact_grad_threshold}, '
                f'lambda_h={compact_lambda_h}')

        # Densification
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.percent_dense = percent_dense
        self.min_opacity = min_opacity
        self.max_screen_size = max_screen_size
        self.sh_update_iter = sh_update_iter

        # scene extent (spatial radius) used by clone/split scale tests
        b = OptimizableCamera.bounds.float()
        self.scene_extent = float(((b[1] - b[0]) / 2).max().item())

    # ------------------------------------------------------------ checkpointing
    @torch.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, *args, **kwargs):
        # support loading a checkpoint with a different number of Gaussians
        pass

    @torch.no_grad()
    def _state_dict_hook(self, module, state_dict, prefix, local_metadata):
        pass

    # --------------------------------------------------------------- densification
    @torch.no_grad()
    def update_gaussians(self, batch: dotdict):
        """Mirror of GaussianTSampler.update_gaussians for the 4D + TGH case.

        Runs on the iteration AFTER the previous render's backward (so
        `scr.grad` is populated). Accumulates view-space gradient stats,
        periodically clones / splits / prunes, and re-assigns the temporal
        hierarchy after any change in Gaussian count.
        """
        if not self.training:
            return
        it = int(batch.meta.iter) if hasattr(batch, 'meta') and 'iter' in batch.meta else 0
        if it <= 0 or it >= self.densify_until_iter:
            return
        if self.last_output is None:
            return

        out = self.last_output
        scr = out.get('scr', None)
        radii = out.get('rad', None)
        active_idx = out.get('gs_idx', None)
        if scr is None or radii is None or active_idx is None or scr.grad is None:
            return
        radii_flat = radii[0] if radii.ndim == 2 else radii          # (M,)

        # accumulate stats only for visible active Gaussians
        visibility = radii_flat > 0
        self.gaussians.add_densification_stats(scr, visibility, active_idx)
        global_vis = active_idx[visibility]
        self.gaussians.max_radii2D[global_vis] = torch.maximum(
            self.gaussians.max_radii2D[global_vis], radii_flat[visibility])

        # progressively reveal higher SH degrees (3DGS convention)
        if it % self.sh_update_iter == 0:
            self.gaussians.oneup_sh_degree()

        # periodic densify / prune
        if it >= self.densify_from_iter and it % self.densification_interval == 0:
            optimizer: Adam = cfg.runner.optimizer
            optimizer_state = dotdict()
            for name, params in self.gaussians.named_parameters():
                if not params.requires_grad:
                    continue
                optimizer_state[params] = dotdict(
                    name=name,
                    old_keep=torch.ones(params.shape[0], dtype=torch.bool, device=params.device),
                    new_keep=torch.ones(params.shape[0], dtype=torch.bool, device=params.device),
                    new_params=None,
                )

            n_clone, n_split, n_prune = self.gaussians.densify_and_prune(
                max_grad=self.densify_grad_threshold,
                min_opacity=self.min_opacity,
                scene_extent=self.scene_extent,
                percent_dense=self.percent_dense,
                max_screen_size=self.max_screen_size,
                optimizer_state=optimizer_state,
            )
            update_optimizer_state(optimizer, optimizer_state)

            # the Gaussian count and per-Gaussian temporal extents both changed,
            # so re-place every Gaussian in the hierarchy
            self.hierarchy.assign(self.gaussians)
            counts = self.hierarchy.level_counts().tolist()
            log(yellow_slim(
                f'[it {it}] densify: +clone {n_clone}, +split {n_split}, '
                f'-prune {n_prune}, now {len(self.gaussians)} Gaussians, '
                f'levels (last=global)={counts}'))

    # ----------------------------------------------------------------- rendering
    def forward(self, batch: dotdict):
        # adaptive control runs first, using the previous iteration's grads
        self.update_gaussians(batch)

        # periodically apply the lambda_h cutoff for the compact-appearance model
        if self.compact_appearance and self.training:
            it = int(batch.meta.get('iter', 0)) if hasattr(batch, 'meta') else 0
            if it > 0 and it % self.compact_check_every == 0:
                prop = self.gaussians.update_compact_appearance(self.compact_lambda_h)
                batch.output.compact_prop = prop

        # normalized timestamp for this view
        _, time = self.sample_index_time(batch)
        t = float(time.reshape(-1)[0].item())

        # Eq. 7: Gaussians active at this timestamp
        mask = self.hierarchy.active_mask(t)
        idx = mask.nonzero(as_tuple=True)[0]

        # Sec. 3.2.1: condition the 4D Gaussians into 3D Gaussians at time t
        xyz3, cov6, occ1 = self.gaussians.get_conditional_3d(t)
        feat = self.gaussians.get_features                            # (N, (deg+1)^2, 3)

        xyz3 = xyz3[idx]
        cov6 = cov6[idx]
        occ1 = occ1[idx]
        feat = feat[idx]

        # view-dependent color via spherical harmonics
        cam_center = (-batch.R[0].mT @ batch.T[0])[..., 0]            # (3,)
        view_dir = normalize(xyz3.detach() - cam_center)              # (M, 3)
        active_deg = int(self.gaussians.active_sh_degree.item())
        sh = feat.mT[..., :(active_deg + 1) ** 2]                     # (M, 3, (deg+1)^2)
        rgb3 = (eval_sh(active_deg, sh, view_dir) + 0.5).clamp(0.0, 1.0)  # (M, 3)

        # rasterize the conditioned 3D Gaussians
        camera = to_x(prepare_gaussian_camera(batch), torch.float)
        rgb, acc, dpt, meta = render_diff_gauss(xyz3, rgb3, cov6, occ1, camera)
        # `meta.scr` is non-leaf (built as `zeros_like(...) + 0` inside the
        # rasterizer); retain its grad so we can read it next iter for
        # densification stats.
        if meta.scr.requires_grad:
            meta.scr.retain_grad()

        # standard EasyVolcap output bookkeeping
        self.store_output(None, xyz3[None], rgb, acc, dpt, batch)
        # retain rasterization byproducts for densification next iter
        batch.output.scr = meta.scr
        batch.output.rad = meta.radii.unsqueeze(0)         # (1, M)
        batch.output.gs_idx = idx                          # global index of each active Gaussian
        batch.output.gs_active = idx.numel()
        self.last_output = batch.output
        return batch.output
