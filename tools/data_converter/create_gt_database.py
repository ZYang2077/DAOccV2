import pickle
import os
from os import path as osp

import mmcv
import numpy as np
from mmcv import track_iter_progress
from mmcv.ops import roi_align
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO

from mmdet3d.core.bbox import box_np_ops as box_np_ops
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.vis_aware_utils import compute_visibility, obb_mask_indices
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from PIL import Image


OCC_FREE_LABELS = {
    "occ3d_nus": 17,
    "occ3d_waymo": 15,
    "surround_occ": 0,
}

OCC_GRID_CONFIG = {
    "occ3d": {
        "range_min": np.array([-40.0, -40.0, -1.0], dtype=np.float64),
        "voxel_size": np.array([0.4, 0.4, 0.4], dtype=np.float64),
        "grid_size": np.array([200, 200, 16], dtype=np.int32),
    },
    "surround_occ": {
        "range_min": np.array([-50.0, -50.0, -5.0], dtype=np.float64),
        "voxel_size": np.array([0.5, 0.5, 0.5], dtype=np.float64),
        "grid_size": np.array([200, 200, 16], dtype=np.int32),
    },
}


def _poly2mask(mask_ann, img_h, img_w):
    if isinstance(mask_ann, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        rle = maskUtils.merge(rles)
    elif isinstance(mask_ann["counts"], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    else:
        # rle
        rle = mask_ann
    mask = maskUtils.decode(rle)
    return mask


def _parse_coco_ann_info(ann_info):
    gt_bboxes = []
    gt_labels = []
    gt_bboxes_ignore = []
    gt_masks_ann = []

    for i, ann in enumerate(ann_info):
        if ann.get("ignore", False):
            continue
        x1, y1, w, h = ann["bbox"]
        if ann["area"] <= 0:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        if ann.get("iscrowd", False):
            gt_bboxes_ignore.append(bbox)
        else:
            gt_bboxes.append(bbox)
            gt_masks_ann.append(ann["segmentation"])

    if gt_bboxes:
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_labels = np.array(gt_labels, dtype=np.int64)
    else:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)

    if gt_bboxes_ignore:
        gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    else:
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

    ann = dict(bboxes=gt_bboxes, bboxes_ignore=gt_bboxes_ignore, masks=gt_masks_ann)

    return ann


def crop_image_patch_v2(pos_proposals, pos_assigned_gt_inds, gt_masks):
    import torch
    from torch.nn.modules.utils import _pair

    device = pos_proposals.device
    num_pos = pos_proposals.size(0)
    fake_inds = torch.arange(num_pos, device=device).to(dtype=pos_proposals.dtype)[
        :, None
    ]
    rois = torch.cat([fake_inds, pos_proposals], dim=1)  # Nx5
    mask_size = _pair(28)
    rois = rois.to(device=device)
    gt_masks_th = (
        torch.from_numpy(gt_masks)
        .to(device)
        .index_select(0, pos_assigned_gt_inds)
        .to(dtype=rois.dtype)
    )
    # Use RoIAlign could apparently accelerate the training (~0.1s/iter)
    targets = roi_align(gt_masks_th, rois, mask_size[::-1], 1.0, 0, True).squeeze(1)
    return targets


def crop_image_patch(pos_proposals, gt_masks, pos_assigned_gt_inds, org_img):
    num_pos = pos_proposals.shape[0]
    masks = []
    img_patches = []
    for i in range(num_pos):
        gt_mask = gt_masks[pos_assigned_gt_inds[i]]
        bbox = pos_proposals[i, :].astype(np.int32)
        x1, y1, x2, y2 = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        mask_patch = gt_mask[y1 : y1 + h, x1 : x1 + w]
        masked_img = gt_mask[..., None] * org_img
        img_patch = masked_img[y1 : y1 + h, x1 : x1 + w]

        img_patches.append(img_patch)
        masks.append(mask_patch)
    return img_patches, masks


def crop_img_patch_with_box(pos_proposals, org_img, waymo=False):
    """Crop all camera patches for every instance.

    Args:
        pos_proposals (list[np.ndarray] | np.ndarray): Per-instance arrays
            with shape ``(num_views, 5)``. An ``(N, 5)`` array is treated as
            the legacy one-view-per-instance representation.
        org_img (Sequence[np.ndarray]): Multi-view images.

    Returns:
        list[list[np.ndarray]]: Patches grouped by 3D instance and ordered in
        the same way as the input views.
    """
    if isinstance(pos_proposals, np.ndarray):
        proposals_per_instance = [row.reshape(1, 5) for row in pos_proposals]
    else:
        proposals_per_instance = [
            np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
            for boxes in pos_proposals
        ]

    patches_per_instance = []
    for instance_boxes in proposals_per_instance:
        instance_patches = []
        for bbox in instance_boxes:
            x1, y1, x2, y2 = bbox[:4].astype(np.int32)
            camera_idx = int(bbox[4])
            img = np.asarray(org_img[camera_idx])
            height, width = img.shape[:2]

            # nuScenes projections are clipped already, but clipping here
            # also makes generated databases robust to rounded coordinates.
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, x1, width - 1))
            y2 = int(np.clip(y2, y1, height - 1))
            instance_patches.append(img[y1:y2 + 1, x1:x2 + 1])
        patches_per_instance.append(instance_patches)

    return patches_per_instance


def _get_image_shape(example, dataset_class_name):
    imgs = example["img"]
    if imgs is None or len(imgs) == 0:
        raise ValueError("Empty or None image list")

    image_shapes = []
    for img in imgs:
        img_arr = np.asarray(img)
        if img_arr.ndim != 3:
            raise ValueError("Invalid image dimension (less than 2D)")
        h, w = img_arr.shape[0], img_arr.shape[1]
        image_shapes.append((h, w))

    if "waymo" in str(dataset_class_name).lower():
        return image_shapes
    else:
        return image_shapes[0]


def _load_surround_occ_semantics(occ_gt_path, grid_size):
    occ = np.load(occ_gt_path)
    occ = occ.astype(np.int32)

    gt = np.zeros(tuple(grid_size.tolist()), dtype=np.int32)
    occ[..., 3][occ[..., 3] == 0] = 255
    coords = occ[:, :3].astype(np.int32)
    gt[coords[:, 0], coords[:, 1], coords[:, 2]] = occ[:, 3]
    return gt


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    with_mask=False,
    load_augmented=None,
    with_occ=False,
    occ_data_type='occ3d',
    visibility_method="dda_nb",
    visibility_depth_eps=0.5,
):
    """Given the raw data, generate the ground truth database.

    Args:
        dataset_class_name （str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str): Path of the info file.
            Default: None.
        mask_anno_path (str): Path of the mask_anno.
            Default: None.
        used_classes (list[str]): Classes have been used.
            Default: None.
        database_save_path (str): Path to save database.
            Default: None.
        db_info_save_path (str): Path to save db_info.
            Default: None.
        relative_path (bool): Whether to use relative path.
            Default: True.
        with_mask (bool): Whether to use mask.
            Default: False.

        with_occ (bool): Whether to use occ.
        occ_data_type (str): 'occ3d', 'surround_occ', 'open_occ'
        visibility_method (str): Visibility algorithm used for precomputing
            occ3d_enriched. Options: 'depth_buffer', 'dda', 'dda_nb'.
            Default: 'dda_nb'.
        visibility_depth_eps (float): Depth tolerance for 'depth_buffer'
            visibility method.
            Default: 0.5.
    """
    print(f"Create GT Database of {dataset_class_name}")
    dataset_cfg = dict(
        type=dataset_class_name, dataset_root=data_path, ann_file=info_path
    )
    if with_mask:
        raise NotImplementedError
    if dataset_class_name == "KittiDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=with_mask,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=4,
                    use_dim=4,
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    elif dataset_class_name in ["NuScenesDataset", "NuScenesDatasetOccupancy"]:
        if not load_augmented:
            dataset_cfg.update(
                use_valid_flag=True,
                modality=dict(
                    use_lidar=True,
                    use_depth=False,
                    use_lidar_intensity=True,
                    use_camera=True,
                ),
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=5,
                        use_dim=5,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        use_dim=[0, 1, 2, 3, 4],
                        pad_empty_sweeps=True,
                        remove_close=True,
                    ),
                    dict(type='LoadMultiViewImageFromFiles'),
                    dict(
                        type="LoadAnnotations3D",
                        with_bbox_3d=True,
                        with_label_3d=True,
                        with_bbox=True,
                        with_label=True,
                    ),
                ],
            )
            if with_occ:
                dataset_cfg['pipeline'].append(
                    dict(
                        type='LoadOccGTFromFile',
                        data_type=occ_data_type,
                    )
                )
                dataset_cfg.update(
                    data_type=occ_data_type,
                )
        else:
            dataset_cfg.update(
                use_valid_flag=True,
                pipeline=[
                    dict(
                        type="LoadPointsFromFile",
                        coord_type="LIDAR",
                        load_dim=16,
                        use_dim=list(range(16)),
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadPointsFromMultiSweeps",
                        sweeps_num=10,
                        load_dim=16,
                        use_dim=list(range(16)),
                        pad_empty_sweeps=True,
                        remove_close=True,
                        load_augmented=load_augmented,
                    ),
                    dict(
                        type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True
                    ),
                ],
            )

    elif dataset_class_name == "WaymoDataset":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=False,
            ),
            pipeline=[
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=6,
                    use_dim=5,
                    tanh_dim=[3, 4],
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                ),
            ],
        )

    elif dataset_class_name == "WaymoDatasetOccupancy":
        dataset_cfg.update(
            test_mode=False,
            split="training",
            modality=dict(
                use_lidar=True,
                use_depth=False,
                use_lidar_intensity=True,
                use_camera=True,
            ),
            load_interval=5,
            pipeline=[
                dict(type='LoadMultiViewImageFromFilesWaymo', image_type='jpg'),
                dict(
                    type="LoadPointsFromFile",
                    coord_type="LIDAR",
                    load_dim=6,
                    use_dim=5,
                    tanh_dim=[3, 4],
                ),
                dict(
                    type="LoadAnnotations3D",
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_bbox=True,
                    with_label=True,
                ),
                dict(
                    type="LoadOccGTFromFileWaymo",
                    data_root="data/waymo/occ3d/voxel04/training/",
                    num_classes=16,
                    free_label=23,
                    use_larger=True,  # TODO: [yz] Keep aligned with parameters defined in the config file
                    crop_x=False,
                    use_infov=True,
                ),
            ],
            pose_file="data/waymo/kitti_format/cam_infos.pkl",  # TODO: [yz] fix hard coding
            occ_gt_data_root="data/waymo/occ3d/voxel04/training/"
        )
    else:
        raise ValueError(
            f"Unsupported dataset class: {dataset_class_name}. "
            f"Currently supported datasets: "
            f"['KittiDataset', 'NuScenesDataset', 'NuScenesDatasetOccupancy', 'WaymoDataset', 'WaymoDatasetOccupancy']"
        )

    dataset = build_dataset(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_dbinfos_train.pkl")
    mmcv.mkdir_or_exist(database_save_path)
    all_db_infos = dict()

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        input_dict = dataset.get_data_info(j)
        dataset.pre_pipeline(input_dict)
        example = dataset.pipeline(input_dict)
        annos = example["ann_info"]
        image_idx = example["sample_idx"]
        points = example["points"].tensor.numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].tensor.numpy()
        names = annos["gt_names"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        gt_boxes = example['gt_bboxes']
        # img_path = osp.split(example['img_info']['filename'][0])[-1]
        img_path = osp.split(example['image_paths'][0])[-1]
        if gt_boxes.shape[0] == 0:
            print(f'skip image {img_path} for empty mask')
            continue
        # h, w = example['img_shape'][:2]
        gt_boxes_multiview = example.get('gt_bboxes_multiview')
        if gt_boxes_multiview is None:
            gt_boxes_multiview = [box.reshape(1, 5) for box in gt_boxes]
        object_img_patches = crop_img_patch_with_box(
            gt_boxes_multiview, example['img'],
            dataset_class_name == "WaymoDatasetOccupancy")

        if with_occ:
            lidar2ego = example.get("lidar2ego", None)
            if lidar2ego is None:
                raise ValueError(f"Missing lidar2ego for sample {image_idx}")
            lidar2ego_mat = np.asarray(lidar2ego, dtype=np.float64)

            lidar2image = example.get("lidar2image", None)
            camera2ego = example.get("camera2ego", None)
            camera_intrinsics = example.get("camera_intrinsics", None)
            img_shape = _get_image_shape(example, dataset_class_name)

            if visibility_method == "depth_buffer":
                raise ValueError("'depth_buffer' method is deprecated, only 'dda_nb' is supported")
            elif visibility_method == "dda":
                raise ValueError("'dda' method is deprecated, only 'dda_nb' is supported")
            elif visibility_method == "dda_nb":
                if camera2ego is None:
                    raise ValueError(
                        f"visibility_method='{visibility_method}' requires camera2ego for sample {image_idx}"
                    )
                if camera_intrinsics is None:
                    raise ValueError(
                        f"visibility_method='{visibility_method}' requires camera_intrinsics for sample {image_idx}"
                    )
                if img_shape is None:
                    raise ValueError(
                        f"visibility_method='{visibility_method}' requires image shape for sample {image_idx}"
                    )
            else:
                raise ValueError(
                    f"Unknown visibility_method: {visibility_method}. "
                    "Supported: 'depth_buffer', 'dda', 'dda_nb'"
                )

            donor_semantics = example['voxel_semantics']
            occ_enriched_per_obj = []
            if occ_data_type == "occ3d":
                if dataset_class_name == "WaymoDatasetOccupancy":
                    occ_free_label = OCC_FREE_LABELS[occ_data_type + '_waymo']  # num class - 1 # TODO: [yz] hard coding
                    occ_coordinate = 'lidar'
                else:
                    occ_free_label = OCC_FREE_LABELS[occ_data_type + '_nus']
                    occ_coordinate = 'ego'
                for i in range(num_obj):
                    # TODO: [yz] obb_indices and obb_indices_expand ?
                    obb_indices = obb_mask_indices(gt_boxes_3d[i],
                                                   occ_coordinate=occ_coordinate,
                                                   lidar2ego_mat=lidar2ego_mat,
                                                   expand_xyz=[0.3, 0.3, 0])
                    ii, jj, kk = obb_indices[:, 0], obb_indices[:, 1], obb_indices[:, 2]
                    labels = donor_semantics[ii, jj, kk].astype(np.int32)

                    vis_score, visible_count, total_count, per_camera_vis = compute_visibility(
                        instance_obb_indices=obb_indices,
                        instance_semantics=donor_semantics,
                        scene_semantics=donor_semantics,
                        lidar2image_list=lidar2image,
                        lidar2ego=lidar2ego_mat,
                        camera2occ_list=camera2ego,
                        cam_intrinsic_list=camera_intrinsics,
                        img_shape=img_shape,
                        ignore_label=occ_free_label,
                        depth_eps=visibility_depth_eps,
                        method=visibility_method,
                    )

                    occ_enriched_per_obj.append(
                        {
                            # "occ_path": occ_dir,
                            # "occ_data_type": occ_data_type,
                            "obb_indices": obb_indices,
                            "obb_labels": labels,
                            "vis_score": vis_score,
                            # "visible_count": visible_count,
                            # "total_count": total_count,
                            # "per_camera_vis": per_camera_vis,
                        }
                    )

            elif occ_data_type == "surround_occ":
                occ_cfg = OCC_GRID_CONFIG[occ_data_type]
                occ_range_min = occ_cfg["range_min"]
                occ_voxel_size = occ_cfg["voxel_size"]
                occ_grid_size = occ_cfg["grid_size"]
                occ_free_label = OCC_FREE_LABELS[occ_data_type]

                camera2lidar = example.get("camera2lidar", None)
                if camera2lidar is None:
                    ego2lidar = np.linalg.inv(lidar2ego_mat)
                    camera2lidar = [ego2lidar @ np.asarray(cam2ego, dtype=np.float64) for cam2ego in camera2ego]
                else:
                    camera2lidar = [np.asarray(cam2lidar, dtype=np.float64) for cam2lidar in camera2lidar]

                for i in range(num_obj):
                    obb_indices = obb_mask_indices(
                        gt_boxes_3d[i],
                        occ_coordinate='lidar',
                        expand_xyz=[0.4, 0.4, 0],
                        range_min=occ_range_min,
                        voxel_size=occ_voxel_size,
                        grid_size=occ_grid_size,
                    )  # TODO: [yz] hard code
                    ii, jj, kk = obb_indices[:, 0], obb_indices[:, 1], obb_indices[:, 2]
                    labels = donor_semantics[ii, jj, kk].astype(np.int32)

                    vis_score, visible_count, total_count, per_camera_vis = compute_visibility(
                        instance_obb_indices=obb_indices,
                        instance_semantics=donor_semantics,
                        scene_semantics=donor_semantics,
                        lidar2image_list=lidar2image,
                        lidar2ego=lidar2ego_mat,
                        camera2occ_list=camera2lidar,
                        cam_intrinsic_list=camera_intrinsics,
                        img_shape=img_shape,
                        ignore_label=occ_free_label,
                        depth_eps=visibility_depth_eps,
                        method=visibility_method,
                        range_min=occ_range_min,
                        voxel_size=occ_voxel_size,
                        grid_size=occ_grid_size,
                    )

                    occ_enriched_per_obj.append(
                        {
                            # "occ_path": occ_path,
                            # "occ_data_type": occ_data_type,
                            "obb_indices": obb_indices,
                            "obb_labels": labels,
                            "vis_score": vis_score,
                            # "visible_count": visible_count,
                            # "total_count": total_count,
                            # "per_camera_vis": per_camera_vis,
                        }
                    )
            else:
                raise NotImplementedError("For now, only support occ3d and surround_occ")

        for i in range(num_obj):
            instance_boxes_2d = np.asarray(
                gt_boxes_multiview[i], dtype=np.float32).reshape(-1, 5)
            if instance_boxes_2d.shape[0] == 0:
                # Keep camera-invisible instances in the info annotations,
                # but do not add unusable entries to the multimodal DB.
                continue

            filename = f"{image_idx}_{names[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]

            areas = ((instance_boxes_2d[:, 2] - instance_boxes_2d[:, 0]) *
                     (instance_boxes_2d[:, 3] - instance_boxes_2d[:, 1]))
            primary_view_idx = int(np.argmax(areas))
            image_patch_paths = []
            for view_idx, (box2d, patch) in enumerate(zip(
                    instance_boxes_2d, object_img_patches[i])):
                camera_idx = int(box2d[4])
                if view_idx == primary_view_idx:
                    abs_patch_path = abs_filepath + '.png'
                    rel_patch_path = rel_filepath + '.png'
                else:
                    suffix = f'.cam{camera_idx}.png'
                    abs_patch_path = abs_filepath + suffix
                    rel_patch_path = rel_filepath + suffix
                if not osp.exists(abs_patch_path):
                    mmcv.imwrite(patch, abs_patch_path)
                image_patch_paths.append(rel_patch_path)
            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                db_info = {
                    "name": names[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    # "occ3d": example["occ3d"],  # TODO: [yz] support other occ dataset
                }
                if with_occ:
                    # occ_enriched_key = "occ3d_enriched" if occ_data_type == "occ3d" else "surround_occ_enriched"
                    db_info[occ_data_type + "_enriched"] = occ_enriched_per_obj[i]
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                db_info.update({
                    "box2d_camera": gt_boxes[i],
                    "box2d_camera_multiview": instance_boxes_2d,
                    "image_patch_paths": image_patch_paths,
                })
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
