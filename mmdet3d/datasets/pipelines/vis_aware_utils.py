import os
import numpy as np
import numba as nb
import open3d as o3d
from PIL import Image

__all__ = ["compute_visibility", "obb_mask_indices", "object_paste_visualize"]

# ============== Object Paste & Visibility-Aware Augmentation Constants ==============
FP_DTYPE = np.float64
IDX_DTYPE = np.int32
# Occ3D-nuScenes grid definition
# TODO: [yz] move hard code param to config
RANGE_MIN = np.array([-40.0, -40.0, -1.0], dtype=FP_DTYPE)
VOXEL_SIZE = np.array([0.4, 0.4, 0.4], dtype=FP_DTYPE)
GRID_SIZE  = np.array([200, 200, 16], dtype=IDX_DTYPE)  # Nx, Ny, Nz
INF = 1e30

# ============== General Functions ==============
def index_to_occ(indices, range_min=RANGE_MIN, voxel_size=VOXEL_SIZE):
    """Convert voxel indices (N,3) to occupancy-frame coordinates (voxel centers)."""
    range_min = np.asarray(range_min, dtype=FP_DTYPE)
    voxel_size = np.asarray(voxel_size, dtype=FP_DTYPE)
    return indices.astype(FP_DTYPE) * voxel_size + range_min + voxel_size / 2


def occ_to_index(xyz_occ, range_min=RANGE_MIN, voxel_size=VOXEL_SIZE):
    """Convert occupancy-frame coordinates to voxel indices (float, not clipped)."""
    range_min = np.asarray(range_min, dtype=FP_DTYPE)
    voxel_size = np.asarray(voxel_size, dtype=FP_DTYPE)
    return (xyz_occ.astype(FP_DTYPE) - range_min) / voxel_size


# ============== Object Paste Functions ==============
def obb_mask_indices(
    lidar_box,
    occ_coordinate='ego',
    lidar2ego_mat=None,
    margin=0,
    expand_xyz=0.0,
    range_min=RANGE_MIN,
    voxel_size=VOXEL_SIZE,
    grid_size=GRID_SIZE,
):
    """
    Returns the voxel index (M,3) inside the OBB, clipped to the legal range.

    Args:
        lidar_box: 3D box in lidar coordinate system, [x, y, z, w, l, h, yaw, ...]
        occ_coordinate: the coordinate system where the true value of occ is located, ego or lidar
        margin: additional voxel index bounds extended outside AABB
        expand_xyz: Expand the box size in continuous space, in meters
                    - Scalar: equal expansion in all directions
                    - (3,) array: expand x/y/z respectively

    illustrate:
        expand_xyz is the "per side" expansion amount, so ultimately dims becomes dims + 2 * expand_xyz
    """
    assert occ_coordinate in ['ego', 'lidar']
    if occ_coordinate == 'ego':
        assert lidar2ego_mat is not None
        center = np.array(
            [lidar_box[0], lidar_box[1], lidar_box[2] + lidar_box[5] / 2.0, 1.0], dtype=np.float64
        )
        center = (lidar2ego_mat @ center)[:3]
        yaw = -float(lidar_box[6]) + np.pi / 2.0
    else:
        center = np.array(
            [lidar_box[0], lidar_box[1], lidar_box[2] + lidar_box[5] / 2.0], dtype=np.float64
        )
        yaw = -float(lidar_box[6])

    center = np.asarray(center, dtype=FP_DTYPE)
    dims_xyz = np.asarray([lidar_box[3], lidar_box[4], lidar_box[5]], dtype=FP_DTYPE)

    if np.isscalar(expand_xyz):
        expand_xyz = np.array([expand_xyz, expand_xyz, expand_xyz], dtype=FP_DTYPE)
    else:
        expand_xyz = np.asarray(expand_xyz, dtype=FP_DTYPE)
        assert expand_xyz.shape == (3,), f"expand_xyz should be scalar or (3,), got {expand_xyz.shape}"

    dims_xyz = dims_xyz + 2.0 * expand_xyz

    cx, cy, cz = center
    dx, dy, dz = dims_xyz
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0

    # --- 1) Construct 8 corner points (±hx, ±hy, ±hz) with OBB local coordinates ---
    corners_local = np.array(
        [[sx * hx, sy * hy, sz * hz]
         for sx in (-1.0, 1.0)
         for sy in (-1.0, 1.0)
         for sz in (-1.0, 1.0)],
        dtype=FP_DTYPE,
    )  # (8,3)

    # --- 2) Rotate to occ_coordinate and translate to the center to get the occ_coordinate coordinates of 8 corner points ---
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=FP_DTYPE,
    )
    corners_occ = corners_local @ Rz.T + np.array([cx, cy, cz], dtype=FP_DTYPE)  # (8,3)

    # --- 3) Find the real AABB (occ_coordinate space) from 8 corner points, then map it to the index space and round it up ---
    xyz_min = corners_occ.min(axis=0)
    xyz_max = corners_occ.max(axis=0)

    idx_min = np.floor(occ_to_index(xyz_min, range_min=range_min, voxel_size=voxel_size)).astype(IDX_DTYPE) - margin
    idx_max = np.ceil(occ_to_index(xyz_max, range_min=range_min, voxel_size=voxel_size)).astype(IDX_DTYPE) + margin

    # --- 4) Crop to legal index range ---
    i0 = max(int(idx_min[0]), 0)
    j0 = max(int(idx_min[1]), 0)
    k0 = max(int(idx_min[2]), 0)
    i1 = min(int(idx_max[0]), grid_size[0] - 1)
    j1 = min(int(idx_max[1]), grid_size[1] - 1)
    k1 = min(int(idx_max[2]), grid_size[2] - 1)

    if i0 > i1 or j0 > j1 or k0 > k1:
        return np.zeros((0, 3), dtype=IDX_DTYPE)

    # --- 5) Enumerate candidate voxels within the real AABB ---
    I = np.arange(i0, i1 + 1, dtype=IDX_DTYPE)
    J = np.arange(j0, j1 + 1, dtype=IDX_DTYPE)
    K = np.arange(k0, k1 + 1, dtype=IDX_DTYPE)
    II, JJ, KK = np.meshgrid(I, J, K, indexing='ij')
    ijk = np.stack([II.ravel(), JJ.ravel(), KK.ravel()], axis=-1)  # (N,3)

    # --- 6) Restore the candidate voxel center to occ_coordinate coordinates ---
    xyz = np.asarray(range_min, dtype=FP_DTYPE) + (ijk.astype(FP_DTYPE) + 0.5) * np.asarray(voxel_size, dtype=FP_DTYPE)  # (N,3)

    # --- 7) Accurate OBB internal determination ---
    c_xyz = np.array([cx, cy, cz], dtype=FP_DTYPE)
    p_local = (xyz - c_xyz) @ Rz  # (N,3)

    inside = (
        (np.abs(p_local[:, 0]) <= hx) &
        (np.abs(p_local[:, 1]) <= hy) &
        (np.abs(p_local[:, 2]) <= hz)
    )

    return ijk[inside].astype(IDX_DTYPE)


def object_paste_visualize(
    data,
    semantics,
    sampled_gt_bboxes_3d=None,
    occ_coordinate='ego',
    dataset_type='nus',
    free_label=17,
    range_min: list = RANGE_MIN,
    voxel_size: list = VOXEL_SIZE,
    offline_visualize=True,
    offline_save_dir="./debug_object_paste_vis_data",
):
    """
    Visualization:
      1) Split visualization into two windows:
         - LiDAR point cloud + 3D boxes
         - Occ3D voxels + 3D boxes
      2) Original GT boxes (blue)
      3) Sampled GT boxes sampled_gt_bboxes_3d (yellow)
      4) LiDAR point cloud from data['points'] (grayscale)

    Args:
      data: contains lidar2ego, gt_bboxes_3d, points, etc.
      semantics: (200,200,16) semantic voxel (EGO grid)
      sampled_gt_bboxes_3d: (M, >=7) boxes in LiDAR frame; not drawn if None
      occ_coordinate: coordinate frame of ground truth occ, supports ['ego', 'lidar']
      dataset_type: 'nus' or 'waymo'
      free_label: label for empty voxels
      range_min: minimum coordinate range of voxel grid, default [-40.0, -40.0, -1.0]
      voxel_size: voxel grid size, default [0.4, 0.4, 0.4]
      offline_visualize: if True, save all data needed by the offline
        visualizer and skip online Open3D windows
      offline_save_dir: root directory used when offline_visualize is True
    """
    assert occ_coordinate in ['ego', 'lidar']
    assert dataset_type in ['nus', 'waymo']
    if dataset_type == 'nus':
        token = str(data['token'])
    else:  # waymo
        token = str(data['sample_idx'])
        assert occ_coordinate == 'lidar'
    # Convert to numpy array for numerical calculation compatibility
    range_min = np.array(range_min, dtype=FP_DTYPE)
    voxel_size = np.array(voxel_size, dtype=FP_DTYPE)

    # 1. Define save path (debug directory only)
    save_dir = "./debug_image_past_vis"
    os.makedirs(os.path.join(save_dir, token), exist_ok=True)

    # 2. Check if valid images
    if 'img' in data and data['img'] is not None and len(data['img']) > 0:
        image_paths = data['image_paths']
        imgs = data['img']

        # 3. Iterate and save all camera images
        for idx, img in enumerate(imgs):
            filename = os.path.basename(image_paths[idx])
            if dataset_type == 'waymo':
                parent_folder = os.path.dirname(image_paths[idx])
                camera_name = os.path.basename(parent_folder)
                filename = camera_name + '_' + filename
            save_path = os.path.join(save_dir, token, filename)

            if isinstance(img, Image.Image):
                img.save(save_path)
            elif isinstance(img, np.ndarray):
                Image.fromarray(img).save(save_path)

        print(f"✅ Debug images saved to directory: {save_dir}")

    # ----------- 1) Voxel Point Cloud (EGO) -----------
    occ_grid = semantics.astype(np.uint8)

    xs, ys, zs = np.indices(occ_grid.shape)
    xs = xs.flatten()
    ys = ys.flatten()
    zs = zs.flatten()
    labels = occ_grid.flatten()
    mask = labels != free_label
    xs, ys, zs, labels = xs[mask], ys[mask], zs[mask], labels[mask]

    coords = np.stack([xs, ys, zs], axis=-1).astype(FP_DTYPE)
    coords_world = range_min + coords * voxel_size + voxel_size / 2.0

    palette_pts = np.array(
        [
            [0, 0, 0, 255],
            [255, 120, 50, 255],  # barrier              orangey
            [255, 192, 203, 255],  # bicycle              pink
            [255, 255, 0, 255],  # bus                  yellow
            [0, 150, 245, 255],  # car                  blue
            [0, 255, 255, 255],  # construction_vehicle cyan
            [200, 180, 0, 255],  # motorcycle           dark orange
            [255, 0, 0, 255],  # pedestrian           red
            [255, 240, 150, 255],  # traffic_cone         light yellow
            [135, 60, 0, 255],  # trailer              brown
            [160, 32, 240, 255],  # truck                purple
            [255, 0, 255, 255],  # driveable_surface    dark pink
            [139, 137, 137, 255],  # other_flat
            [75, 0, 75, 255],  # sidewalk             dard purple
            [150, 240, 80, 255],  # terrain              light green
            [230, 230, 250, 255],  # manmade              white
            [0, 175, 0, 255],  # vegetation           green
            [0, 0, 255, 255],  # ego car
            [255, 99, 71, 255],
            [0, 191, 255, 255]
        ],
        dtype=np.uint8,
    )
    palette_pts = palette_pts[:, :3]
    colors = palette_pts[labels % len(palette_pts)] / 255.0

    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(coords_world)
    voxel_pcd.colors = o3d.utility.Vector3dVector(colors)

    # ----------- 2) LiDAR -> EGO Transform -----------
    lidar2ego = np.array(data["lidar2ego"], dtype=FP_DTYPE)

    # ----------- 3) LiDAR Point Cloud -----------
    lidar_geoms = []
    num_lidar_points = 0
    points_xyz = np.zeros((0, 3), dtype=FP_DTYPE)
    points_data = data.get('points', None)
    if points_data is not None:
        if hasattr(points_data, 'tensor'):
            points_data = points_data.tensor
        if hasattr(points_data, 'detach'):
            points_data = points_data.detach()
        if hasattr(points_data, 'cpu'):
            points_data = points_data.cpu()
        if hasattr(points_data, 'numpy'):
            points_np = points_data.numpy()
        else:
            points_np = np.asarray(points_data)

        if points_np.ndim == 2 and points_np.shape[0] > 0 and points_np.shape[1] >= 3:
            points_xyz = points_np[:, :3].astype(FP_DTYPE)
            if occ_coordinate == 'ego':
                points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=FP_DTYPE)], axis=-1)
                points_xyz = (points_h @ lidar2ego.T)[:, :3]

            lidar_pcd = o3d.geometry.PointCloud()
            lidar_pcd.points = o3d.utility.Vector3dVector(points_xyz)
            lidar_colors = np.full((points_xyz.shape[0], 3), 0.7, dtype=FP_DTYPE)
            lidar_pcd.colors = o3d.utility.Vector3dVector(lidar_colors)
            lidar_geoms.append(lidar_pcd)
            num_lidar_points = points_xyz.shape[0]

    # ----------- 4) 3D Boxes -----------
    gt_boxes = data['gt_bboxes_3d'].tensor.numpy()
    n_sampled = len(sampled_gt_bboxes_3d) if sampled_gt_bboxes_3d is not None else 0
    original_gt_boxes = gt_boxes[:-n_sampled] if n_sampled > 0 else gt_boxes
    sampled_boxes = sampled_gt_bboxes_3d if sampled_gt_bboxes_3d is not None else []

    def _box_center_yaw_extent(box):
        cx, cy, cz, dx, dy, dz, yaw = box[:7]
        if occ_coordinate == 'ego':
            center_lidar = np.array([cx, cy, cz + dz / 2.0, 1.0], dtype=FP_DTYPE)
            center = (lidar2ego @ center_lidar)[:3]
            yaw_vis = -yaw + np.pi / 2
        else:
            center = np.array([cx, cy, cz + dz / 2.0], dtype=FP_DTYPE)
            yaw_vis = -yaw

        return center, yaw_vis, np.array([dx, dy, dz], dtype=FP_DTYPE)

    def _build_obb(box):
        center, yaw_vis, extent = _box_center_yaw_extent(box)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw_vis])
        return o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)

    def _boxes_to_corners(boxes):
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 8, 3), dtype=FP_DTYPE)

        corners_local_norm = np.array(
            [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
            ],
            dtype=FP_DTYPE,
        )
        corners = []
        for box in boxes:
            center, yaw_vis, extent = _box_center_yaw_extent(box)
            c, s = np.cos(yaw_vis), np.sin(yaw_vis)
            rot = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=FP_DTYPE,
            )
            corners.append((corners_local_norm * extent) @ rot.T + center)
        return np.asarray(corners, dtype=FP_DTYPE)

    def _append_boxes(geoms):
        for box in original_gt_boxes:
            obb = _build_obb(box)
            obb.color = (0.0, 0.0, 1.0)  # blue
            geoms.append(obb)

        for box in sampled_boxes:
            obb = _build_obb(box)
            obb.color = (1.0, 1.0, 0.0)  # yellow
            geoms.append(obb)

    voxel_geoms = [voxel_pcd]
    _append_boxes(voxel_geoms)
    _append_boxes(lidar_geoms)

    print(
        f"✅ Voxels: {len(coords_world)} | LiDAR points: {num_lidar_points} | GT boxes: {len(gt_boxes)} | "
        f"Sampled: {0 if sampled_gt_bboxes_3d is None else len(sampled_gt_bboxes_3d)}"
    )

    if offline_visualize:
        offline_token_dir = os.path.join(offline_save_dir, token)
        os.makedirs(offline_token_dir, exist_ok=True)
        offline_path = os.path.join(offline_token_dir, "object_paste_vis_data.npz")
        np.savez_compressed(
            offline_path,
            token=np.array(token),
            semantics=occ_grid,
            points_xyz=np.asarray(points_xyz if num_lidar_points > 0 else np.zeros((0, 3)), dtype=FP_DTYPE),
            original_box_corners=_boxes_to_corners(original_gt_boxes),
            sampled_box_corners=_boxes_to_corners(sampled_boxes),
            gt_boxes=np.asarray(gt_boxes, dtype=FP_DTYPE),
            sampled_boxes=np.asarray(sampled_boxes, dtype=FP_DTYPE),
            occ_coordinate=np.array(occ_coordinate),
            dataset_type=np.array(dataset_type),
            free_label=np.array(free_label, dtype=np.int32),
            range_min=range_min,
            voxel_size=voxel_size,
            palette=palette_pts.astype(np.uint8),
        )
        print(f"✅ Offline visualization data saved to: {offline_path}")
        return

    def visualize_with_options(geoms, window_name, point_size=None, bg_color=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        opt = vis.get_render_option()
        if point_size is not None:
            opt.point_size = point_size
        if bg_color is not None:
            opt.background_color = np.array(bg_color)

        for geom in geoms:
            vis.add_geometry(geom)

        vis.run()
        vis.destroy_window()

    if len(lidar_geoms) > 0:
        visualize_with_options(
            lidar_geoms,
            window_name=token + " | LiDAR Points (Gray) | 3D Boxes",
            point_size=2.0,
            bg_color=[0.0, 0.0, 0.0]
        )
    visualize_with_options(
        voxel_geoms,
        window_name=token + " | Voxels | 3D Boxes"
    )


# ============== Visibility-Aware Augmentation Functions ==============

@nb.njit(cache=True)
def ray_box_intersection_nb(ray_o, ray_d, box_min, box_max):
    """
    Ray-AABB intersection for numba.
    Returns: (hit, t_enter, t_exit)
    """
    t_enter = -INF
    t_exit = INF

    for i in range(3):
        if abs(ray_d[i]) < 1e-12:
            if ray_o[i] < box_min[i] or ray_o[i] > box_max[i]:
                return False, 0.0, 0.0
            continue

        t1 = (box_min[i] - ray_o[i]) / ray_d[i]
        t2 = (box_max[i] - ray_o[i]) / ray_d[i]

        if t1 < t2:
            t_near = t1
            t_far = t2
        else:
            t_near = t2
            t_far = t1

        if t_near > t_enter:
            t_enter = t_near
        if t_far < t_exit:
            t_exit = t_far

        if t_enter > t_exit:
            return False, 0.0, 0.0

    if t_exit < 0.0:
        return False, 0.0, 0.0

    return True, t_enter, t_exit


@nb.njit(cache=True)
def trace_ray_dda_nb(start_idx, end_idx, external_occupied, grid_size, eps=1e-6):
    """
    Standard 3D DDA ray traversal in voxel-index space.
    Returns True if ray reaches target voxel without hitting any external occupied voxel.
    """
    ray_o = np.empty(3, dtype=np.float64)
    ray_d = np.empty(3, dtype=np.float64)

    for i in range(3):
        ray_o[i] = start_idx[i]
        ray_d[i] = end_idx[i] - start_idx[i]

    ray_len = np.sqrt(ray_d[0] * ray_d[0] + ray_d[1] * ray_d[1] + ray_d[2] * ray_d[2])
    if ray_len < 1e-8:
        return True

    for i in range(3):
        ray_d[i] /= ray_len

    box_min = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    box_max = np.array(
        [float(grid_size[0]), float(grid_size[1]), float(grid_size[2])],
        dtype=np.float64,
    )

    hit, t_enter, t_exit = ray_box_intersection_nb(ray_o, ray_d, box_min, box_max)
    if not hit:
        return False

    t = t_enter
    if t < 0.0:
        t = 0.0
    t += eps
    if t > t_exit:
        return False

    p0 = ray_o[0] + t * ray_d[0]
    p1 = ray_o[1] + t * ray_d[1]
    p2 = ray_o[2] + t * ray_d[2]

    current = np.empty(3, dtype=np.int64)
    current[0] = int(np.floor(p0))
    current[1] = int(np.floor(p1))
    current[2] = int(np.floor(p2))

    target = np.empty(3, dtype=np.int64)
    target[0] = int(np.floor(end_idx[0]))
    target[1] = int(np.floor(end_idx[1]))
    target[2] = int(np.floor(end_idx[2]))

    if target[0] < 0 or target[0] >= grid_size[0]:
        return False
    if target[1] < 0 or target[1] >= grid_size[1]:
        return False
    if target[2] < 0 or target[2] >= grid_size[2]:
        return False

    step = np.zeros(3, dtype=np.int64)
    if ray_d[0] > 0.0:
        step[0] = 1
    elif ray_d[0] < 0.0:
        step[0] = -1

    if ray_d[1] > 0.0:
        step[1] = 1
    elif ray_d[1] < 0.0:
        step[1] = -1

    if ray_d[2] > 0.0:
        step[2] = 1
    elif ray_d[2] < 0.0:
        step[2] = -1

    t_max = np.empty(3, dtype=np.float64)
    t_delta = np.empty(3, dtype=np.float64)

    for axis in range(3):
        if abs(ray_d[axis]) < 1e-12:
            t_max[axis] = INF
            t_delta[axis] = INF
        else:
            if step[axis] > 0:
                next_boundary = float(current[axis] + 1)
            else:
                next_boundary = float(current[axis])

            if axis == 0:
                t_max[axis] = t + (next_boundary - p0) / ray_d[axis]
            elif axis == 1:
                t_max[axis] = t + (next_boundary - p1) / ray_d[axis]
            else:
                t_max[axis] = t + (next_boundary - p2) / ray_d[axis]

            t_delta[axis] = 1.0 / abs(ray_d[axis])

    max_steps = int(grid_size[0] + grid_size[1] + grid_size[2]) * 4

    for _ in range(max_steps):
        if current[0] < 0 or current[0] >= grid_size[0]:
            return False
        if current[1] < 0 or current[1] >= grid_size[1]:
            return False
        if current[2] < 0 or current[2] >= grid_size[2]:
            return False

        if current[0] == target[0] and current[1] == target[1] and current[2] == target[2]:
            return True

        if external_occupied[current[0], current[1], current[2]]:
            return False

        axis = 0
        if t_max[1] < t_max[axis]:
            axis = 1
        if t_max[2] < t_max[axis]:
            axis = 2

        if t_max[axis] > t_exit + eps:
            return False

        current[axis] += step[axis]
        t = t_max[axis]
        t_max[axis] += t_delta[axis]

    return False


@nb.njit(cache=True)
def ray_visibility_mask_nb(cam_origin_idx, candidate_voxel_idx, external_occupied, grid_size):
    """
    Batch ray casting for one camera.

    Args:
        cam_origin_idx: (3,) float64, camera origin in voxel-index space
        candidate_voxel_idx: (N, 3) float64, voxel centers in voxel-index space
        external_occupied: bool grid
        grid_size: (3,) int64

    Returns:
        visible_mask: (N,) bool
    """
    n = candidate_voxel_idx.shape[0]
    visible_mask = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        end_idx = candidate_voxel_idx[i]
        if trace_ray_dda_nb(cam_origin_idx, end_idx, external_occupied, grid_size):
            visible_mask[i] = True

    return visible_mask


def occ_to_image_points(pts_occ, camera2occ, cam_intrinsic):
    """
    Project occ points to image plane using camera extrinsic/intrinsic.

    Args:
        pts_occ: (N, 3)
        camera2occ: (4, 4), camera-to-occ transform
        cam_intrinsic: (3, 3) or (4, 4)

    Returns:
        uv: (N, 2)
        depth: (N,)
        valid_mask: (N,) depth > 0
    """
    occ2cam = np.linalg.inv(np.asarray(camera2occ, dtype=FP_DTYPE))

    pts_occ_h = np.concatenate(
        [np.asarray(pts_occ, dtype=FP_DTYPE), np.ones((len(pts_occ), 1), dtype=FP_DTYPE)],
        axis=-1,
    )  # (N,4)

    pts_cam_h = pts_occ_h @ occ2cam.T
    pts_cam = pts_cam_h[:, :3]

    cam_intrinsic = np.asarray(cam_intrinsic, dtype=FP_DTYPE)
    intrinsic = cam_intrinsic[:3, :3] if cam_intrinsic.shape[0] == 4 else cam_intrinsic
    proj = (intrinsic @ pts_cam.T).T  # (N,3)

    # For standard pinhole K, proj[:, 2] is camera z. Waymo cam_infos use a
    # KITTI-style projection whose third row selects forward x after the
    # camera-to-lidar transform, so depth must come from the projection itself.
    depth = proj[:, 2]
    valid_mask = depth > 1e-6
    uv = proj[:, :2] / np.maximum(depth[:, None], 1e-6)
    return uv, depth, valid_mask


def filter_voxel_candidates_by_fov(
    voxel_centers_occ,
    camera2occ,
    cam_intrinsic,
    img_shape,
):
    """
    Keep only voxels whose centers fall inside the camera image FOV.
    """
    uv, depth, valid_mask = occ_to_image_points(voxel_centers_occ, camera2occ, cam_intrinsic)
    H, W = img_shape[:2]

    in_fov = (
        valid_mask &
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    return in_fov


def compute_instance_visibility_dda_nb(
    instance_obb_indices,
    instance_semantics,
    scene_semantics,
    camera2occ_list,
    cam_intrinsic_list=None,
    img_shape=None,
    ignore_label=17,
    range_min=RANGE_MIN,
    voxel_size=VOXEL_SIZE,
    grid_size=GRID_SIZE,
):
    """
    Visibility with:
    1) per-camera FOV candidate filtering
    2) numba-accelerated DDA ray casting
    3) union visibility across cameras
    """
    if instance_obb_indices.shape[0] == 0:
        return 0.0, 0, 0, {}

    grid_size = np.asarray(grid_size, dtype=IDX_DTYPE)

    ii = instance_obb_indices[:, 0]
    jj = instance_obb_indices[:, 1]
    kk = instance_obb_indices[:, 2]

    ii_clip = np.clip(ii, 0, grid_size[0] - 1)
    jj_clip = np.clip(jj, 0, grid_size[1] - 1)
    kk_clip = np.clip(kk, 0, grid_size[2] - 1)

    instance_labels = instance_semantics[ii_clip, jj_clip, kk_clip]
    nonempty_mask = instance_labels != ignore_label
    nonempty_indices = instance_obb_indices[nonempty_mask]

    total_count = nonempty_indices.shape[0]
    if total_count == 0:
        return 0.0, 0, 0, {}

    instance_mask = np.zeros(grid_size, dtype=np.bool_)
    inst_i = ii_clip[nonempty_mask]
    inst_j = jj_clip[nonempty_mask]
    inst_k = kk_clip[nonempty_mask]
    instance_mask[inst_i, inst_j, inst_k] = True

    external_occupied = (scene_semantics != ignore_label) & (~instance_mask)
    external_occupied = np.ascontiguousarray(external_occupied.astype(np.bool_))

    voxel_centers_idx = np.ascontiguousarray(nonempty_indices.astype(FP_DTYPE) + 0.5)
    voxel_centers_occ = index_to_occ(
        nonempty_indices, range_min=range_min, voxel_size=voxel_size
    ).astype(FP_DTYPE)

    visible_union = np.zeros(total_count, dtype=np.bool_)
    per_camera_vis = {}

    num_cams = len(camera2occ_list)
    if cam_intrinsic_list is not None and len(cam_intrinsic_list) != num_cams:
        raise ValueError(
            f"camera2occ_list and cam_intrinsic_list length mismatch: "
            f"{num_cams} vs {len(cam_intrinsic_list)}"
        )

    for cam_idx, camera2occ in enumerate(camera2occ_list):
        cam_origin_occ = np.asarray(camera2occ[:3, 3], dtype=FP_DTYPE)
        cam_origin_idx = occ_to_index(
            cam_origin_occ, range_min=range_min, voxel_size=voxel_size
        ).astype(FP_DTYPE)

        if cam_intrinsic_list is not None and img_shape is not None:
            if isinstance(img_shape, tuple):  # occ3d-nus
                img_shape_cur = img_shape
            elif isinstance(img_shape, list) and isinstance(img_shape[0], tuple):  # occ3d-waymo
                img_shape_cur = img_shape[cam_idx]
            else:
                raise ValueError("Invalid image shape")
            cam_intrinsic = cam_intrinsic_list[cam_idx]
            in_fov = filter_voxel_candidates_by_fov(
                voxel_centers_occ=voxel_centers_occ,
                camera2occ=camera2occ,
                cam_intrinsic=cam_intrinsic,
                img_shape=img_shape_cur,
            )
        else:
            in_fov = np.ones(total_count, dtype=np.bool_)

        pending_mask = (~visible_union) & in_fov
        candidate_indices = np.where(pending_mask)[0]

        if candidate_indices.size == 0:
            per_camera_vis[cam_idx] = 0
            continue

        candidate_voxel_idx = np.ascontiguousarray(voxel_centers_idx[candidate_indices])

        cam_visible_mask = ray_visibility_mask_nb(
            cam_origin_idx,
            candidate_voxel_idx,
            external_occupied,
            grid_size,
        )

        cam_visible_count = int(cam_visible_mask.sum())
        per_camera_vis[cam_idx] = cam_visible_count

        visible_ids = candidate_indices[cam_visible_mask]
        visible_union[visible_ids] = True

        if visible_union.all():
            for rest_cam in range(cam_idx + 1, num_cams):
                per_camera_vis[rest_cam] = 0
            break

    visible_count = int(visible_union.sum())
    vis_score = visible_count / total_count if total_count > 0 else 0.0

    return vis_score, visible_count, total_count, per_camera_vis


def compute_visibility(
    instance_obb_indices,
    instance_semantics,
    scene_semantics,
    lidar2image_list=None,
    lidar2ego=None,
    camera2occ_list=None,
    cam_intrinsic_list=None,
    img_shape=None,
    ignore_label=17,
    depth_eps=0.5,
    method='dda_nb',
    range_min=RANGE_MIN,
    voxel_size=VOXEL_SIZE,
    grid_size=GRID_SIZE,
):
    """
    Unified visibility computation function with configurable method.

    Args:
        instance_obb_indices: (M, 3) voxel indices inside the instance's OBB
        instance_semantics: (200, 200, 16) donor frame semantics (to filter non-empty)
        scene_semantics: (200, 200, 16) current scene semantics (for occlusion check)
        lidar2image_list: List of 6 (4, 4) LiDAR-to-image projection matrices
                          (required for 'depth_buffer' method)
        lidar2ego: (4, 4) LiDAR-to-EGO transform matrix (required for 'depth_buffer' method)
        camera2occ_list: List of 6 (4, 4) camera-to-occ transform matrices
                         (required for 'dda' method)
        cam_intrinsic_list: List of 6 camera intrinsic matrices (3x3 or 4x4),
                            used for per-camera FOV filtering in 'dda_nb' method
        img_shape: (H, W) image dimensions (required for FOV filtering)
        ignore_label: Label value for empty/unknown voxels (default: 17)
        depth_eps: Depth tolerance in meters for occlusion check (default: 0.5)
                   (only used by 'depth_buffer' method)
        method: Visibility computation method, one of:
                - 'depth_buffer': 2D image depth buffer projection (original method)
                - 'dda': 3D DDA ray tracing in occupancy grid (new method)
                - 'dda_nb': Numba 3D DDA for better performance

    Returns:
        vis_score: float in [0, 1], ratio of visible voxels
        visible_voxel_count: int, number of visible voxels
        total_voxel_count: int, total non-empty voxels in instance
        per_camera_visibility: dict mapping camera_idx -> visible count in that camera
    """
    if method == 'depth_buffer':
        raise ValueError("'depth_buffer' method is deprecated, only 'dda_nb' is supported")
    elif method == 'dda':
        raise ValueError("'dda' method is deprecated, only 'dda_nb' is supported")
    elif method == 'dda_nb':
        if camera2occ_list is None:
            raise ValueError("'dda_nb' method requires camera2occ_list")
        return compute_instance_visibility_dda_nb(
            instance_obb_indices=instance_obb_indices,
            instance_semantics=instance_semantics,
            scene_semantics=scene_semantics,
            camera2occ_list=camera2occ_list,
            cam_intrinsic_list=cam_intrinsic_list,
            img_shape=img_shape,
            ignore_label=ignore_label,
            range_min=range_min,
            voxel_size=voxel_size,
            grid_size=grid_size,
        )
    else:
        raise ValueError(
            f"Unknown visibility method: {method}. "
            f"Supported methods: 'depth_buffer', 'dda', 'dda_nb'"
        )


# ============== End Visibility-Aware Augmentation ==============
