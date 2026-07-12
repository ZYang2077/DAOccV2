from typing import Tuple, List

import torch
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from torch import nn
import torch.nn.functional as F
import einops
from mmdet3d.models.builder import VTRANSFORMS
from ..utils.reference_poinst import get_reference_points

import warnings

__all__ = ["BEVTransformV2", "BEVTransformV3", "BEVDenoiseTransform"]


@VTRANSFORMS.register_module()
class BEVTransformV2(BaseModule):
    """
    Follow SparseBEV and optimize the bilinear interpolation sampling process.
    """
    def __init__(
            self,
            x: List[float], y: List[float], z: List[float],
            xs: int, ys: int, zs: int,
            input_size: List[int],
            in_channels: int = 256,
            out_channels: int = 128,
            top_type: str = 'lidar',
            down_sample: bool = False,
            down_sample_scale: int = 2,
            down_sample_channels: List[int] = [128*10, 64*10, 32*10, 16*10]
    ):
        """
        Args:
            x, y, z: Range in three directions.
            xs, ys, zs: Resolution in three directions.
            input_size (tuple(int)): Size of input images in format of (height, width).
            in_channels (int): Channels of input feature.
            out_channels (int): Channels of transformed feature.
            top_type (str): Coordinate system in which the task is labeled, 'lidar' or 'ego',
                        eg: 'lidar' for 3D detection and 'ego' for Occ3D-nuScenes.
        """
        super(BEVTransformV2, self).__init__()
        self.pc_range = [x[0], y[0], z[0], x[1], y[1], z[1]]
        self.volume_size = [int(s) for s in [xs, ys, zs]]
        ref_3d = get_reference_points(self.pc_range[0], self.pc_range[3],
                                      self.pc_range[1], self.pc_range[4],
                                      self.pc_range[2], self.pc_range[5],
                                      self.volume_size[0], self.volume_size[1], self.volume_size[2])
        self.register_buffer('ref_3d', ref_3d)
        if in_channels != out_channels:
            self.transfer_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.transfer_conv = None
        self.out_channels = out_channels
        self.initial_flag = True
        self.input_size = input_size
        self.top_type = top_type
        if down_sample:
            assert down_sample_scale in [1, 2], 'only support 1x or 2x downsampling!'
            if down_sample_scale == 2:
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True),
                    nn.Conv2d(
                        down_sample_channels[1],
                        down_sample_channels[2],
                        3,
                        stride=down_sample_scale,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(down_sample_channels[2]),
                    nn.ReLU(True),
                    nn.Conv2d(down_sample_channels[2], down_sample_channels[3], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[3]),
                    nn.ReLU(True),
                )
            else:  # down_sample_scale == 1
                assert len(down_sample_channels) == 2, "the len of 'down_sample_channels: List[int]' should be 2!"
                self.down_sample = nn.Sequential(
                    nn.Conv2d(down_sample_channels[0], down_sample_channels[1], 3, padding=1, bias=False),
                    nn.BatchNorm2d(down_sample_channels[1]),
                    nn.ReLU(True)
                )
        else:
            self.down_sample = None
        self.fp16_enabled = False  # TODO: important!!! # will be automatically set as true

    @force_fp32()
    def forward(self,
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                **kwargs,
                ):
        """Transform image-view feature into bird-eye-view feature.

        Args:
        Returns:
        """
        B = x.shape[0]
        if self.transfer_conv is not None:
            x = einops.rearrange(x, 'B N C H W -> (B N) C H W')
            x = self.transfer_conv(x)
            x = einops.rearrange(x, '(B N) C H W -> B C N H W', B=B)
        else:
            x = einops.rearrange(x, 'B N C H W -> B C N H W')

        if self.top_type == 'ego':
            assert 'camera_ego2global' in kwargs
            camera_ego2global = kwargs['camera_ego2global']
            keyego2global = camera_ego2global[:, 0, ...].unsqueeze(1)  # (B, 1, 4, 4)
            global2keyego = torch.inverse(keyego2global.double())  # (B, 1, 4, 4)
            camera2sensor = global2keyego @ camera_ego2global.double() @ camera2ego.double()  # (B, N_views, 4, 4)
            camera2sensor = camera2sensor.float()
        elif self.top_type == 'lidar':
            camera2sensor = camera2lidar
        else:
            raise NotImplementedError
        reference_points_img, volume_mask = self.point_sampling(camera2sensor, camera_intrinsics[..., :3, :3],
                                                                img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
                                                                lidar_aug_matrix)

        bs, num_points, _ = reference_points_img.shape  # num_points: 324000
        valid_index_per_batch = [volume_mask[i].squeeze(-1).nonzero().squeeze(-1) for i in range(bs)]  # [torch.Size([312387])]
        max_len = max([len(per_batch) for per_batch in valid_index_per_batch])

        reference_points_rebatch = reference_points_img.new_zeros([bs, max_len, 3])
        for j in range(bs):
            reference_points_rebatch[j, :len(valid_index_per_batch[j])] = reference_points_img[j, valid_index_per_batch[j]]
        reference_points_rebatch = reference_points_rebatch.view(bs, max_len, 1, 1, 3)
        reference_points_rebatch = 2 * reference_points_rebatch - 1  # torch.Size([1, 312387, 1, 1, 3])

        sampling_feats = F.grid_sample(
            x,
            reference_points_rebatch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        sampling_feats = einops.rearrange(sampling_feats,
                                          'b c max_len 1 1 -> b max_len c',
                                          b=bs)
        feats_volume = x.new_zeros([bs, num_points, self.out_channels])
        for j in range(bs):
            feats_volume[j, valid_index_per_batch[j]] = sampling_feats[j, :len(valid_index_per_batch[j])]

        feats_volume = einops.rearrange(feats_volume,
                                        'bs (z h w) c -> bs (z c) w h',
                                        z=self.volume_size[2], h=self.volume_size[0], w=self.volume_size[1])
        if self.down_sample is not None:
            feats_volume = self.down_sample(feats_volume)
        return feats_volume

    @force_fp32()
    def point_sampling(self, camera2sensor, cam2imgs, post_rots, post_trans, bda, mode='fix'):
        B, N, _, _ = camera2sensor.shape
        num_points = self.ref_3d.shape[0]

        # [(NP 3) -> (1 NP 3)] - [(B 3) -> (B 1 3)] -> (B NP 3)
        reference_points = self.ref_3d.view(1, num_points, 3) - bda[:, :3, 3].view(B, 1, 3)
        # [(B 3 3) -> (B 1 3 3)] @ [(B NP 3) -> (B NP 3 1)] -> (B NP 3 1)
        reference_points = torch.inverse(bda[:, :3, :3].view(B, 1, 3, 3)).matmul(reference_points.unsqueeze(-1))
        # (B NP 3 1) -> (B 1 NP 3 1) -> (B 1 NP 3)
        reference_points = reference_points.view(B, 1, num_points, 3, 1).squeeze(-1)
        # (B 1 NP 3) - (B N 1 3) -> (B N NP 3) -> (B N NP 3 1)
        reference_points = (reference_points - camera2sensor[:, :, :3, 3].view(B, N, 1, 3)).unsqueeze(-1)
        # (B N 3 3)
        combine = cam2imgs.matmul(camera2sensor[:, :, :3, :3].transpose(-1, -2))
        # [(B N 3 3) -> (B N 1 3 3)] @ (B N NP 3 1) -> (B N NP 3 1) -> (B N NP 3)
        reference_points_img = combine.view(B, N, 1, 3, 3).matmul(reference_points).squeeze(-1)

        eps = 1e-5
        # (B N NP 1)
        volume_mask = (reference_points_img[..., 2:3] > eps)
        # (B N NP 2)
        reference_points_img = reference_points_img[..., 0:2] / torch.maximum(
            reference_points_img[..., 2:3], torch.ones_like(reference_points_img[..., 2:3]) * eps)

        # do post-transformation
        post_rots2 = post_rots[:, :, :2, :2]
        post_trans2 = post_trans[:, :, :2]
        # [(B N 2 2) -> (B N 1 2 2)] @ [(B N NP 2) -> (B N NP 2 1)] -> (B N NP 2 1)
        reference_points_img = post_rots2.view(B, N, 1, 2, 2).matmul(reference_points_img.unsqueeze(-1))
        # [(B N NP 2 1) -> (B N NP 2)] + [(B N 2) -> (B N 1 2)] -> (B N NP 2)
        reference_points_img = reference_points_img.squeeze(-1) + post_trans2.view(B, N, 1, 2)

        H_in, W_in = self.input_size
        reference_points_img[..., 0] /= W_in  # 1600 w
        reference_points_img[..., 1] /= H_in  # 928 h

        volume_mask = (volume_mask & (reference_points_img[..., 1:2] > 0.0)
                       & (reference_points_img[..., 1:2] < 1.0)
                       & (reference_points_img[..., 0:1] < 1.0)
                       & (reference_points_img[..., 0:1] > 0.0))

        volume_mask = torch.nan_to_num(volume_mask)

        # (B N NP 2) -> (B NP N 2)
        reference_points_img = reference_points_img.permute(0, 2, 1, 3)
        # (B N NP 1) -> (B NP N 1)
        volume_mask = volume_mask.permute(0, 2, 1, 3)
        if mode == 'fix':
            idx_camera = torch.argmax(volume_mask.squeeze(-1).float(), dim=-1)
        elif mode == 'random':
            raise NotImplementedError
        else:
            raise NotImplementedError

        idx_batch = torch.arange(B, dtype=torch.long, device=reference_points_img.device)
        idx_point = torch.arange(num_points, dtype=torch.long, device=reference_points_img.device)
        idx_batch = idx_batch.view(B, 1).expand(B, num_points)
        idx_point = idx_point.view(1, num_points).expand(B, num_points)
        reference_points_img = reference_points_img[idx_batch, idx_point, idx_camera]
        volume_mask = volume_mask[idx_batch, idx_point, idx_camera]

        coors_camera = idx_camera[..., None].float() / (N - 1)
        reference_points_img = torch.cat([reference_points_img, coors_camera], dim=-1)

        return reference_points_img, volume_mask


@VTRANSFORMS.register_module()
class BEVTransformV3(BEVTransformV2):
    """BEVTransformV2 with optional non-empty voxel masking.

    Adds a function ``generate_nonempty_mask`` that produces a boolean mask
    over ``self.ref_3d`` indicating which pre-defined Lidar-frame voxels
    correspond to non-empty (and optionally camera-visible) voxels in the
    ground-truth occupancy grid.  When ``use_nonempty_mask=True``, this mask
    is intersected with ``volume_mask`` during forward so that only non-empty
    voxels participate in image sampling.

    Extra init args:
        use_nonempty_mask (bool): Whether to call ``generate_nonempty_mask``
            inside forward and apply it to ``volume_mask``.  Default ``False``.
        occ_range (list[float]): [xmin, ymin, zmin, xmax, ymax, zmax] of the
            occupancy grid in ego coordinates.  Required when
            ``use_nonempty_mask=True``.
        free_label (int): Label id that represents free/empty in
            ``voxel_semantics``.  Default ``17``.
        mask_type (str): ``'camera_visible'`` – non-empty AND camera-visible;
            ``'nonempty'`` – non-empty only.  Default ``'camera_visible'``.
    """

    def __init__(
        self,
        x: List[float], y: List[float], z: List[float],
        xs: int, ys: int, zs: int,
        input_size: List[int],
        in_channels: int = 256,
        out_channels: int = 128,
        top_type: str = 'lidar',
        down_sample: bool = False,
        down_sample_scale: int = 2,
        down_sample_channels: List[int] = [128 * 10, 64 * 10, 32 * 10, 16 * 10],
        use_nonempty_mask: bool = False,
        occ_range: List[float] = None,
        free_label: int = 17,
        mask_type: str = 'camera_visible',
    ):
        warnings.warn("BEVTransformV3 is only maintained for compatibility purposes. \
                       Please use BEVDenoiseTransform instead.",)
        super().__init__(
            x=x, y=y, z=z, xs=xs, ys=ys, zs=zs,
            input_size=input_size, in_channels=in_channels,
            out_channels=out_channels, top_type=top_type,
            down_sample=down_sample, down_sample_scale=down_sample_scale,
            down_sample_channels=down_sample_channels,
        )
        self.use_nonempty_mask = use_nonempty_mask
        self.occ_range = occ_range
        self.free_label = free_label
        assert mask_type in ('nonempty', 'camera_visible')
        self.mask_type = mask_type

        # Pre-compute ref_3d grid spacing for fast nearest-neighbour lookup.
        if occ_range is not None:
            self._init_ref_grid_params()

    def _init_ref_grid_params(self):
        """Cache ref_3d grid parameters for voxel index computation."""
        self._ref_x_min = self.pc_range[0]
        self._ref_y_min = self.pc_range[1]
        self._ref_z_min = self.pc_range[2]
        self._ref_x_max = self.pc_range[3]
        self._ref_y_max = self.pc_range[4]
        self._ref_z_max = self.pc_range[5]
        self._ref_xs = self.volume_size[0]  # x resolution
        self._ref_ys = self.volume_size[1]  # y resolution
        self._ref_zs = self.volume_size[2]  # z resolution
        self._ref_dx = (self._ref_x_max - self._ref_x_min) / self._ref_xs
        self._ref_dy = (self._ref_y_max - self._ref_y_min) / self._ref_ys
        self._ref_dz = (self._ref_z_max - self._ref_z_min) / self._ref_zs

    # ------------------------------------------------------------------
    #  Function A: generate_nonempty_mask
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_nonempty_mask(
            self,
            voxel_semantics: torch.Tensor,
            lidar2ego: torch.Tensor,
            mask_camera: torch.Tensor = None,
            occ_range: List[float] = None,
            free_label: int = None,
            mask_type: str = None,
    ) -> torch.Tensor:
        """Generate a boolean mask over ``self.ref_3d`` marking non-empty voxels.

        Args:
            voxel_semantics (Tensor): (N, dx, dy, dz) – occupancy GT labels.
            lidar2ego (Tensor): (N, 4, 4) – transformation from Lidar to ego.
            mask_camera (Tensor | None): (N, dx, dy, dz) – 0/1 visibility mask.
            occ_range (list[float] | None): [xmin, ymin, zmin, xmax, ymax, zmax].
                Falls back to ``self.occ_range`` when ``None``.
            free_label (int | None): Falls back to ``self.free_label``.
            mask_type (str | None): ``'nonempty'`` or ``'camera_visible'``.
                Falls back to ``self.mask_type``.

        Returns:
            Tensor: Bool mask of shape
                (N, self.volume_size[0], self.volume_size[1], self.volume_size[2])
                – ``True`` means the corresponding ``ref_3d`` voxel is non-empty
                (and camera-visible, if ``mask_type='camera_visible'``).
        """
        if occ_range is None:
            occ_range = self.occ_range
        assert occ_range is not None, 'occ_range must be provided'
        if free_label is None:
            free_label = self.free_label
        if mask_type is None:
            mask_type = self.mask_type

        # Lazily initialise grid params when occ_range was not set at __init__
        if not hasattr(self, '_ref_dx'):
            self._init_ref_grid_params()

        device = voxel_semantics.device
        N = voxel_semantics.shape[0]
        dx, dy, dz = voxel_semantics.shape[1], voxel_semantics.shape[2], voxel_semantics.shape[3]
        occ_xmin, occ_ymin, occ_zmin, occ_xmax, occ_ymax, occ_zmax = occ_range

        # Step 1: non-empty voxel mask in the occ grid
        nonempty = (voxel_semantics != free_label)  # (N, dx, dy, dz)
        if mask_type == 'camera_visible':
            assert mask_camera is not None, \
                "mask_camera is required when mask_type='camera_visible'"
            nonempty = nonempty & (mask_camera.bool())

        # Step 2: map non-empty voxels to ego-frame centre coordinates
        occ_dx = (occ_xmax - occ_xmin) / dx
        occ_dy = (occ_ymax - occ_ymin) / dy
        occ_dz = (occ_zmax - occ_zmin) / dz

        nz = nonempty.nonzero(as_tuple=False)  # (M, 4): batch, ix, iy, iz
        batch_idx = nz[:, 0]
        ix, iy, iz = nz[:, 1], nz[:, 2], nz[:, 3]

        cx_ego = occ_xmin + (ix.float() + 0.5) * occ_dx
        cy_ego = occ_ymin + (iy.float() + 0.5) * occ_dy
        cz_ego = occ_zmin + (iz.float() + 0.5) * occ_dz
        pts_ego = torch.stack([cx_ego, cy_ego, cz_ego], dim=-1)  # (M, 3)

        # Step 3: ego -> lidar via ego2lidar = inv(lidar2ego)
        ego2lidar_rot = torch.inverse(lidar2ego[:, :3, :3])  # (N, 3, 3)
        ego2lidar_trans = lidar2ego[:, :3, 3]  # (N, 3)

        rot = ego2lidar_rot[batch_idx]   # (M, 3, 3)
        trans = ego2lidar_trans[batch_idx]  # (M, 3)
        pts_lidar = torch.bmm(rot, (pts_ego - trans).unsqueeze(-1)).squeeze(-1)  # (M, 3)

        # Step 4: quantise into the ref_3d grid (nearest-neighbour)
        ref_ix = torch.floor(
            (pts_lidar[:, 0] - self._ref_x_min) / self._ref_dx
        ).long().clamp(0, self._ref_xs - 1)
        ref_iy = torch.floor(
            (pts_lidar[:, 1] - self._ref_y_min) / self._ref_dy
        ).long().clamp(0, self._ref_ys - 1)
        ref_iz = torch.floor(
            (pts_lidar[:, 2] - self._ref_z_min) / self._ref_dz
        ).long().clamp(0, self._ref_zs - 1)

        result_mask = voxel_semantics.new_zeros(
            (N, self._ref_xs, self._ref_ys, self._ref_zs), dtype=torch.bool
        )
        result_mask[batch_idx, ref_ix, ref_iy, ref_iz] = True

        return result_mask  # (N, xs, ys, zs)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    @force_fp32()
    def forward(self,
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                **kwargs,
                ):
        B = x.shape[0]
        if self.transfer_conv is not None:
            x = einops.rearrange(x, 'B N C H W -> (B N) C H W')
            x = self.transfer_conv(x)
            x = einops.rearrange(x, '(B N) C H W -> B C N H W', B=B)
        else:
            x = einops.rearrange(x, 'B N C H W -> B C N H W')

        if self.top_type == 'ego':
            assert 'camera_ego2global' in kwargs
            camera_ego2global = kwargs['camera_ego2global']
            keyego2global = camera_ego2global[:, 0, ...].unsqueeze(1)
            global2keyego = torch.inverse(keyego2global.double())
            camera2sensor = global2keyego @ camera_ego2global.double() @ camera2ego.double()
            camera2sensor = camera2sensor.float()
        elif self.top_type == 'lidar':
            camera2sensor = camera2lidar
        else:
            raise NotImplementedError
        reference_points_img, volume_mask = self.point_sampling(
            camera2sensor, camera_intrinsics[..., :3, :3],
            img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
            lidar_aug_matrix)

        # ---------- apply non-empty voxel mask if enabled -----------------
        if self.use_nonempty_mask:
            voxel_semantics = kwargs.get('voxel_semantics', None)
            mask_camera = kwargs.get('mask_camera', None)
            assert voxel_semantics is not None, \
                'voxel_semantics must be in kwargs when use_nonempty_mask=True'

            nonempty_mask = self.generate_nonempty_mask(
                voxel_semantics=voxel_semantics,
                lidar2ego=lidar2ego,
                mask_camera=mask_camera,
            )
            # nonempty_mask: (N, xs, ys, zs)
            # Flatten to (N, num_points, 1) matching volume_mask layout.
            # ref_3d is ordered as [zs*ys*xs, xyz] so
            # flatten must follow the same (zs, ys, xs) -> num_points order.
            nonempty_mask_flat = einops.rearrange(
                nonempty_mask, 'N xs ys zs -> N (zs ys xs) 1'
            ).to(volume_mask.dtype)
            volume_mask = volume_mask * nonempty_mask_flat

        bs, num_points, _ = reference_points_img.shape
        valid_index_per_batch = [
            volume_mask[i].squeeze(-1).nonzero().squeeze(-1) for i in range(bs)
        ]
        max_len = max([len(per_batch) for per_batch in valid_index_per_batch])

        reference_points_rebatch = reference_points_img.new_zeros([bs, max_len, 3])
        for j in range(bs):
            reference_points_rebatch[j, :len(valid_index_per_batch[j])] = \
                reference_points_img[j, valid_index_per_batch[j]]
        reference_points_rebatch = reference_points_rebatch.view(bs, max_len, 1, 1, 3)
        reference_points_rebatch = 2 * reference_points_rebatch - 1

        sampling_feats = F.grid_sample(
            x,
            reference_points_rebatch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        sampling_feats = einops.rearrange(sampling_feats,
                                          'b c max_len 1 1 -> b max_len c',
                                          b=bs)
        feats_volume = x.new_zeros([bs, num_points, self.out_channels])
        for j in range(bs):
            feats_volume[j, valid_index_per_batch[j]] = \
                sampling_feats[j, :len(valid_index_per_batch[j])]

        feats_volume = einops.rearrange(feats_volume,
                                        'bs (z h w) c -> bs (z c) w h',
                                        z=self.volume_size[2],
                                        h=self.volume_size[0],
                                        w=self.volume_size[1])
        if self.down_sample is not None:
            feats_volume = self.down_sample(feats_volume)
        return feats_volume


@VTRANSFORMS.register_module()
class BEVDenoiseTransform(BEVTransformV2):
    """BEVTransformV2 with configurable voxel masking strategies.

    Supports multiple mask types to control which voxels participate in image
    sampling during forward pass.

    Extra init args:
        occ_range (list[float]): [xmin, ymin, zmin, xmax, ymax, zmax] of the
            occupancy grid in ego coordinates. Required for GT-based masks.
        free_label (int): Label id that represents free/empty in
            ``voxel_semantics``. Default ``17``.
        mask_type (str): Mask type to use:
            - ``'none'``: No masking, use all camera-visible voxels.
            - ``'gt_nonempty'``: Non-empty voxels from GT occupancy.
            - ``'gt_nonempty_cam'``: Non-empty AND camera-visible voxels.
            - ``'gt_cam'``: Camera-visible voxels only (from mask_camera).
            - ``'voxel_mask'``: Use voxelized point cloud mask.
            Default ``'none'``.
    """

    def __init__(
        self,
        x: List[float], y: List[float], z: List[float],
        xs: int, ys: int, zs: int,
        input_size: List[int],
        in_channels: int = 256,
        out_channels: int = 128,
        top_type: str = 'lidar',
        down_sample: bool = False,
        down_sample_scale: int = 2,
        down_sample_channels: List[int] = [128 * 10, 64 * 10, 32 * 10, 16 * 10],
        occ_range: List[float] = None,
        free_label: int = 17,
        mask_type: str = 'none',
        compress_z: bool = False,
    ):
        super().__init__(
            x=x, y=y, z=z, xs=xs, ys=ys, zs=zs,
            input_size=input_size, in_channels=in_channels,
            out_channels=out_channels, top_type=top_type,
            down_sample=down_sample, down_sample_scale=down_sample_scale,
            down_sample_channels=down_sample_channels,
        )
        self.occ_range = occ_range
        self.free_label = free_label
        assert mask_type in ('none', 'gt_nonempty', 'gt_nonempty_cam', 'gt_cam', 'voxel_mask')
        self.mask_type = mask_type

        self.compress_z = compress_z
        if self.compress_z:
            assert self.volume_size[2] % 2 == 0, \
                f"compress_z=True requires z to be even, got z={self.volume_size[2]}"

        # Pre-compute ref_3d grid spacing for fast nearest-neighbour lookup.
        if mask_type != 'none':
            self._init_ref_grid_params()

    def _init_ref_grid_params(self):
        """Cache ref_3d grid parameters for voxel index computation."""
        self._ref_x_min = self.pc_range[0]
        self._ref_y_min = self.pc_range[1]
        self._ref_z_min = self.pc_range[2]
        self._ref_x_max = self.pc_range[3]
        self._ref_y_max = self.pc_range[4]
        self._ref_z_max = self.pc_range[5]
        self._ref_xs = self.volume_size[0]  # x resolution
        self._ref_ys = self.volume_size[1]  # y resolution
        self._ref_zs = self.volume_size[2]  # z resolution
        self._ref_dx = (self._ref_x_max - self._ref_x_min) / self._ref_xs
        self._ref_dy = (self._ref_y_max - self._ref_y_min) / self._ref_ys
        self._ref_dz = (self._ref_z_max - self._ref_z_min) / self._ref_zs

    # ------------------------------------------------------------------
    #  Function: generate_mask_from_gt
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_mask_from_gt(
            self,
            lidar2ego: torch.Tensor,
            voxel_semantics: torch.Tensor = None,
            mask_camera: torch.Tensor = None,
            occ_range: List[float] = None,
            free_label: int = None,
            use_nonempty: bool = True,
            use_camera_mask: bool = False,
    ) -> torch.Tensor:
        """Generate a boolean mask over ``self.ref_3d`` from GT occupancy/camera mask.

        Args:
            lidar2ego (Tensor): (N, 4, 4) – transformation from Lidar to ego.
            voxel_semantics (Tensor | None): (N, dx, dy, dz) – occupancy GT labels.
                Required when ``use_nonempty=True``.
            mask_camera (Tensor | None): (N, dx, dy, dz) – 0/1 visibility mask.
                Required when ``use_camera_mask=True`` or when ``use_nonempty=False``.
            occ_range (list[float] | None): [xmin, ymin, zmin, xmax, ymax, zmax].
                Falls back to ``self.occ_range`` when ``None``.
            free_label (int | None): Falls back to ``self.free_label``.
            use_nonempty (bool): If True, use non-empty voxels from voxel_semantics.
            use_camera_mask (bool): If True, apply camera visibility mask.

        Returns:
            Tensor: Bool mask of shape
                (N, self.volume_size[0], self.volume_size[1], self.volume_size[2])
                – ``True`` means the corresponding ``ref_3d`` voxel is selected.
        """
        if occ_range is None:
            occ_range = self.occ_range
        assert occ_range is not None, 'occ_range must be provided'
        if free_label is None:
            free_label = self.free_label

        # Lazily initialise grid params when occ_range was not set at __init__
        if not hasattr(self, '_ref_dx'):
            self._init_ref_grid_params()

        # Determine the source tensor for shape and device
        if voxel_semantics is not None:
            source_tensor = voxel_semantics
        else:
            assert mask_camera is not None, \
                'Either voxel_semantics or mask_camera must be provided'
            source_tensor = mask_camera

        device = source_tensor.device
        N = source_tensor.shape[0]
        dx, dy, dz = source_tensor.shape[1], source_tensor.shape[2], source_tensor.shape[3]
        occ_xmin, occ_ymin, occ_zmin, occ_xmax, occ_ymax, occ_zmax = occ_range

        # Step 1: build the mask based on flags
        if use_nonempty:
            assert voxel_semantics is not None, \
                'voxel_semantics is required when use_nonempty=True'
            mask = (voxel_semantics != free_label)  # (N, dx, dy, dz)
            if use_camera_mask:
                assert mask_camera is not None, \
                    'mask_camera is required when use_camera_mask=True'
                mask = mask & mask_camera.bool()
        else:
            assert mask_camera is not None, \
                'mask_camera is required when use_nonempty=False'
            mask = mask_camera.bool()  # (N, dx, dy, dz)

        # Step 2: map selected voxels to ego-frame centre coordinates
        occ_dx = (occ_xmax - occ_xmin) / dx
        occ_dy = (occ_ymax - occ_ymin) / dy
        occ_dz = (occ_zmax - occ_zmin) / dz

        nz = mask.nonzero(as_tuple=False)  # (M, 4): batch, ix, iy, iz
        batch_idx = nz[:, 0]
        ix, iy, iz = nz[:, 1], nz[:, 2], nz[:, 3]

        cx_ego = occ_xmin + (ix.float() + 0.5) * occ_dx
        cy_ego = occ_ymin + (iy.float() + 0.5) * occ_dy
        cz_ego = occ_zmin + (iz.float() + 0.5) * occ_dz
        pts_ego = torch.stack([cx_ego, cy_ego, cz_ego], dim=-1)  # (M, 3)

        # Step 3: ego -> lidar via ego2lidar = inv(lidar2ego)
        ego2lidar_rot = torch.inverse(lidar2ego[:, :3, :3])  # (N, 3, 3)
        ego2lidar_trans = lidar2ego[:, :3, 3]  # (N, 3)

        rot = ego2lidar_rot[batch_idx]   # (M, 3, 3)
        trans = ego2lidar_trans[batch_idx]  # (M, 3)
        pts_lidar = torch.bmm(rot, (pts_ego - trans).unsqueeze(-1)).squeeze(-1)  # (M, 3)

        # Step 4: quantise into the ref_3d grid (nearest-neighbour)
        ref_ix = torch.floor(
            (pts_lidar[:, 0] - self._ref_x_min) / self._ref_dx
        ).long().clamp(0, self._ref_xs - 1)
        ref_iy = torch.floor(
            (pts_lidar[:, 1] - self._ref_y_min) / self._ref_dy
        ).long().clamp(0, self._ref_ys - 1)
        ref_iz = torch.floor(
            (pts_lidar[:, 2] - self._ref_z_min) / self._ref_dz
        ).long().clamp(0, self._ref_zs - 1)

        result_mask = source_tensor.new_zeros(
            (N, self._ref_xs, self._ref_ys, self._ref_zs), dtype=torch.bool
        )
        result_mask[batch_idx, ref_ix, ref_iy, ref_iz] = True

        return result_mask  # (N, xs, ys, zs)

    # ------------------------------------------------------------------
    #  Function: generate_voxel_mask
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_voxel_mask(
            self,
            voxel_coords_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Generate a boolean mask from voxelized point cloud coordinates.

        Args:
            voxel_coords_list (list[Tensor]): List of length batch_size, each
                element is a tensor of shape (M_i, 3) containing voxel indices
                [dx, dy, dz] for that batch.

        Returns:
            Tensor: Bool mask of shape
                (N, self.volume_size[0], self.volume_size[1], self.volume_size[2])
                – ``True`` means the corresponding voxel is non-empty.
        """
        N = len(voxel_coords_list)
        X, Y, Z = self._ref_xs, self._ref_ys, self._ref_zs
        device = voxel_coords_list[0].device

        result_mask = torch.zeros((N, X, Y, Z), dtype=torch.bool, device=device)

        for batch_idx, voxel_indices in enumerate(voxel_coords_list):
            if voxel_indices.numel() == 0:
                continue

            # Extract three dimension indices (dx, dy, dz)
            dx = voxel_indices[:, 0]
            dy = voxel_indices[:, 1]
            dz = voxel_indices[:, 2]

            # Filter invalid indices (prevent out-of-bounds)
            valid_mask = (dx >= 0) & (dx < X) & (dy >= 0) & (dy < Y) & (dz >= 0) & (dz < Z)
            valid_dx = dx[valid_mask].long()
            valid_dy = dy[valid_mask].long()
            valid_dz = dz[valid_mask].long()

            # Set valid positions to True
            result_mask[batch_idx, valid_dx, valid_dy, valid_dz] = True

        return result_mask  # (N, xs, ys, zs)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    @force_fp32()
    def forward(self,
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                **kwargs,
                ):
        B = x.shape[0]
        if self.transfer_conv is not None:
            x = einops.rearrange(x, 'B N C H W -> (B N) C H W')
            x = self.transfer_conv(x)
            x = einops.rearrange(x, '(B N) C H W -> B C N H W', B=B)
        else:
            x = einops.rearrange(x, 'B N C H W -> B C N H W')

        if self.top_type == 'ego':
            assert 'camera_ego2global' in kwargs
            camera_ego2global = kwargs['camera_ego2global']
            keyego2global = camera_ego2global[:, 0, ...].unsqueeze(1)
            global2keyego = torch.inverse(keyego2global.double())
            camera2sensor = global2keyego @ camera_ego2global.double() @ camera2ego.double()
            camera2sensor = camera2sensor.float()
        elif self.top_type == 'lidar':
            camera2sensor = camera2lidar
        else:
            raise NotImplementedError
        reference_points_img, volume_mask = self.point_sampling(
            camera2sensor, camera_intrinsics[..., :3, :3],
            img_aug_matrix[..., :3, :3], img_aug_matrix[..., :3, 3],
            lidar_aug_matrix)

        # ---------- apply voxel mask based on mask_type -----------------
        if self.mask_type != 'none':
            if self.mask_type == 'gt_nonempty':
                voxel_semantics = kwargs.get('voxel_semantics', None)
                assert voxel_semantics is not None, \
                    'voxel_semantics must be in kwargs when mask_type=gt_nonempty'
                denoise_mask = self.generate_mask_from_gt(
                    lidar2ego=lidar2ego,
                    voxel_semantics=voxel_semantics,
                    use_nonempty=True,
                    use_camera_mask=False,
                )
            elif self.mask_type == 'gt_nonempty_cam':
                voxel_semantics = kwargs.get('voxel_semantics', None)
                mask_camera = kwargs.get('mask_camera', None)
                assert voxel_semantics is not None, \
                    'voxel_semantics must be in kwargs when mask_type=gt_nonempty_cam'
                assert mask_camera is not None, \
                    'mask_camera must be in kwargs when mask_type=gt_nonempty_cam'
                denoise_mask = self.generate_mask_from_gt(
                    lidar2ego=lidar2ego,
                    voxel_semantics=voxel_semantics,
                    mask_camera=mask_camera,
                    use_nonempty=True,
                    use_camera_mask=True,
                )
            elif self.mask_type == 'gt_cam':
                mask_camera = kwargs.get('mask_camera', None)
                assert mask_camera is not None, \
                    'mask_camera must be in kwargs when mask_type=gt_cam'
                denoise_mask = self.generate_mask_from_gt(
                    lidar2ego=lidar2ego,
                    mask_camera=mask_camera,
                    use_nonempty=False,
                )
            elif self.mask_type == 'voxel_mask':
                voxel_coords_list = kwargs.get('voxel_coords_list', None)
                assert voxel_coords_list is not None, \
                    'voxel_coords_list must be in kwargs when mask_type=voxel_mask'
                denoise_mask = self.generate_voxel_mask(voxel_coords_list)
            else:
                raise ValueError(f"Unknown mask_type: {self.mask_type}")

            # denoise_mask: (N, xs, ys, zs)
            # Flatten to (N, num_points, 1) matching volume_mask layout.
            # ref_3d is ordered as [zs*ys*xs, xyz] so
            # flatten must follow the same (zs, ys, xs) -> num_points order.
            nonempty_mask_flat = einops.rearrange(
                denoise_mask, 'N xs ys zs -> N (zs ys xs) 1'
            ).to(volume_mask.dtype)
            volume_mask = volume_mask * nonempty_mask_flat

        bs, num_points, _ = reference_points_img.shape
        valid_index_per_batch = [
            volume_mask[i].squeeze(-1).nonzero().squeeze(-1) for i in range(bs)
        ]
        max_len = max([len(per_batch) for per_batch in valid_index_per_batch])

        reference_points_rebatch = reference_points_img.new_zeros([bs, max_len, 3])
        for j in range(bs):
            reference_points_rebatch[j, :len(valid_index_per_batch[j])] = \
                reference_points_img[j, valid_index_per_batch[j]]
        reference_points_rebatch = reference_points_rebatch.view(bs, max_len, 1, 1, 3)
        reference_points_rebatch = 2 * reference_points_rebatch - 1

        sampling_feats = F.grid_sample(
            x,
            reference_points_rebatch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        sampling_feats = einops.rearrange(sampling_feats,
                                          'b c max_len 1 1 -> b max_len c',
                                          b=bs)
        feats_volume = x.new_zeros([bs, num_points, self.out_channels])
        for j in range(bs):
            feats_volume[j, valid_index_per_batch[j]] = \
                sampling_feats[j, :len(valid_index_per_batch[j])]

        if self.compress_z:
            # z = self.volume_size[2]
            # h = self.volume_size[0]
            # w = self.volume_size[1]
            # c = feats_volume.shape[-1]
            #
            # # (bs, z*h*w, c) -> (bs, z, h, w, c)
            # feats_volume = einops.rearrange(
            #     feats_volume,
            #     'bs (z h w) c -> bs z h w c',
            #     z=z, h=h, w=w
            # )
            #
            # z_half = z // 2
            # first_half = feats_volume[:, :z_half].sum(dim=1)    # (bs, h, w, c)
            # second_half = feats_volume[:, z_half:].sum(dim=1)   # (bs, h, w, c)
            #
            # # (bs, 2, h, w, c)
            # feats_volume = torch.stack([first_half, second_half], dim=1)
            #
            # # -> (bs, 2c, w, h)
            # feats_volume = einops.rearrange(
            #     feats_volume,
            #     'bs two h w c -> bs (two c) w h'
            # )

            z = self.volume_size[2]
            h = self.volume_size[0]
            w = self.volume_size[1]

            # (bs, z*h*w, c) -> (bs, z, h, w, c)
            feats_volume = einops.rearrange(
                feats_volume,
                'bs (z h w) c -> bs z h w c',
                z=z, h=h, w=w
            )
            # (bs, z, h, w, c) -> (bs, h, w, c)
            feats_volume = feats_volume.sum(dim=1)
            # -> (bs, c, w, h)
            feats_volume = einops.rearrange(
                feats_volume,
                'bs h w c -> bs c w h'
            )
        else:
            feats_volume = einops.rearrange(
                feats_volume,
                'bs (z h w) c -> bs (z c) w h',  # z x y -> z y x
                z=self.volume_size[2],
                h=self.volume_size[0],
                w=self.volume_size[1]
            )

        if self.down_sample is not None:
            feats_volume = self.down_sample(feats_volume)
        return feats_volume
