import copy
import os

import mmcv
import numpy as np
import numba as nb
import cv2
from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets import PIPELINES
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines.utils import box_collision_test


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(
        self, sampled_list, name=None, epoch=None, shuffle=True, drop_reminder=False
    ):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx :].copy()
            self._reset()
        else:
            ret = self._indices[self._idx : self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@OBJECTSAMPLERS.register_module()
class DataBaseSampler:
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        dataset_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(
        self,
        info_path,
        dataset_root,
        rate,
        prepare,
        sample_groups,
        classes=None,
        points_loader=dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=4,
            use_dim=[0, 1, 2, 3],
        ),
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)

        db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger

        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f"load {len(v)} {k} database infos")
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info("After filter database:")
        for k, v in db_infos.items():
            logger.info(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos if info["difficulty"] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self.sample_classes, self.sample_max_nums
        ):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(
                max_sample_num - np.sum([n == class_label for n in gt_labels])
            )
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(
                    class_name, sampled_num, avoid_coll_boxes
                )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = (
                    os.path.join(self.dataset_root, info["path"])
                    if self.dataset_root
                    else info["path"]
                )
                results = dict(lidar_path=file_path)
                s_points = self.points_loader(results)["points"]
                s_points.translate(info["box3d_lidar"][:3])

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array(
                [self.cat2label[s["name"]] for s in sampled], dtype=np.long
            )
            ret = {
                "gt_labels_3d": gt_labels,
                "gt_bboxes_3d": sampled_gt_bboxes,
                "points": s_points_list[0].cat(s_points_list),
                "group_ids": np.arange(
                    gt_bboxes.shape[0], gt_bboxes.shape[0] + len(sampled)
                ),
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples


@OBJECTSAMPLERS.register_module()
class MMDataBaseSamplerV2(DataBaseSampler):

    def __init__(self,
                 info_path,
                 dataset_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 check_2D_collision=False,
                 collision_thr=0,
                 collision_in_classes=False,
                 depth_consistent=False,
                 blending_type=None,
                 mixup=1.0,
                 img_num=6,
                 img_loader=dict(type='LoadImageFromFile'),
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     load_dim=4,
                     coord_type='LIDAR',
                     use_dim=[0, 1, 2, 3]),
                 visibility_aware_aug=None):
        super(MMDataBaseSamplerV2, self).__init__(
            info_path=info_path,
            dataset_root=dataset_root,
            rate=rate,
            prepare=prepare,
            sample_groups=sample_groups,
            classes=classes,
            points_loader=points_loader)

        self.blending_type = blending_type
        self.depth_consistent = depth_consistent
        self.check_2D_collision = check_2D_collision
        self.collision_thr = collision_thr
        self.collision_in_classes = collision_in_classes
        self.img_loader = mmcv.build_from_cfg(img_loader, PIPELINES)
        self.mixup = mixup
        self.img_num = img_num

        # Visibility-aware augmentation config
        # self.vis_cfg = visibility_aware_aug or {}
        # self.vis_enabled = self.vis_cfg.get('enabled', False)

    @staticmethod
    def _normalize_multiview_boxes(primary_boxes, multiview_boxes=None):
        """Return one ``(Vi, 5)`` float array per 3D instance."""
        if multiview_boxes is None:
            if primary_boxes is None:
                return None
            return [
                np.asarray(box, dtype=np.float32).reshape(1, 5)
                for box in np.asarray(primary_boxes)
            ]
        normalized = [
            np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
            for boxes in multiview_boxes
        ]
        if primary_boxes is not None and len(normalized) != len(primary_boxes):
            raise ValueError(
                "gt_bboxes_2d_multiview must have one entry per 3D instance: "
                f"got {len(normalized)} and {len(primary_boxes)}")
        return normalized

    @classmethod
    def _db_multiview_boxes(cls, info):
        boxes = info.get('box2d_camera_multiview')
        if boxes is None:
            box = info.get('box2d_camera')
            if box is None:
                return np.empty((0, 5), dtype=np.float32)
            boxes = np.asarray(box, dtype=np.float32).reshape(1, 5)
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 5)

    @staticmethod
    def _primary_box(info, multiview_boxes):
        box = info.get('box2d_camera')
        if box is not None:
            return np.asarray(box, dtype=np.float32).reshape(5)
        if len(multiview_boxes) == 0:
            return None
        areas = ((multiview_boxes[:, 2] - multiview_boxes[:, 0]) *
                 (multiview_boxes[:, 3] - multiview_boxes[:, 1]))
        return multiview_boxes[int(np.argmax(areas))]

    def sample_all(self,
                   gt_bboxes_3d,
                   gt_names,
                   gt_bboxes_2d=None,
                   gt_bboxes_2d_multiview=None,
                   img=None,
                   img_filename=None,
                   lidar2image=None,
                   lidar2ego=None,
                   camera2ego=None,
                   scene_semantics=None,
                   img_shape=None,
                   occ_data_type=None):
        """
        Compatible with 3D-only and 2D+3D modes:
        - When img / gt_bboxes_2d is empty: only do point cloud, 3D box, and occ sampling
        - When img/gt_bboxes_2d is non-empty: perform additional image pasting
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            sampled_num = int(max_sample_num -
                              np.sum([n == class_name for n in gt_names]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes_3d = []
        sampled_gt_bboxes_2d = []
        sampled_gt_bboxes_2d_multiview = []

        avoid_coll_boxes_3d = gt_bboxes_3d
        gt_bboxes_2d_multiview = self._normalize_multiview_boxes(
            gt_bboxes_2d, gt_bboxes_2d_multiview)
        avoid_coll_boxes_2d = gt_bboxes_2d_multiview

        if self.check_2D_collision and gt_bboxes_2d is None:
            raise ValueError(
                "check_2D_collision=True requires gt_bboxes_2d to be provided."
            )

        use_img_branch = (img is not None) and (gt_bboxes_2d is not None)
        origin_img = [im.copy() for im in img] if use_img_branch else None

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num <= 0:
                continue

            sampled_cls = self.sample_class_vis_aware(
                class_name, sampled_num, avoid_coll_boxes_3d,
                avoid_coll_boxes_2d)

            sampled += sampled_cls
            if len(sampled_cls) == 0:
                continue

            sampled_gt_box_3d = np.stack(
                [s['box3d_lidar'] for s in sampled_cls], axis=0)
            sampled_multiview = [
                self._db_multiview_boxes(s) for s in sampled_cls
            ]
            sampled_primary = [
                self._primary_box(s, boxes)
                for s, boxes in zip(sampled_cls, sampled_multiview)
            ]
            sampled_gt_box_2d = None
            if all(box is not None for box in sampled_primary):
                sampled_gt_box_2d = np.stack(sampled_primary, axis=0)

            sampled_gt_bboxes_3d += [sampled_gt_box_3d]
            if use_img_branch and sampled_gt_box_2d is not None:
                sampled_gt_bboxes_2d += [sampled_gt_box_2d]
                sampled_gt_bboxes_2d_multiview.extend(sampled_multiview)

            if self.collision_in_classes:
                # TODO: check whether check collision check among classes is necessary
                avoid_coll_boxes_3d = np.concatenate(
                    [avoid_coll_boxes_3d, sampled_gt_box_3d], axis=0)
                if use_img_branch and sampled_gt_box_2d is not None:
                    avoid_coll_boxes_2d = (
                        avoid_coll_boxes_2d + sampled_multiview)

        if len(sampled) == 0:
            return None

        sampled_gt_bboxes_3d = np.concatenate(sampled_gt_bboxes_3d, axis=0)

        # 3D-only also retains occ/points information
        point_list = []
        occ_list = []

        for idx in range(len(sampled)):
            info = sampled[idx]
            pcd_file_path = os.path.join(self.dataset_root, info['path']) if self.dataset_root else info['path']
            results = dict(lidar_path=pcd_file_path)
            s_points = self.points_loader(results)['points']
            s_points.translate(info['box3d_lidar'][:3])
            point_list.append(s_points)

            # Precomputed in db_info during GT database generation.
            occ_enriched = info.get(occ_data_type + "_enriched", None)
            if occ_enriched is None:
                raise ValueError(
                    f"Missing '{occ_data_type}'_enriched for sampled object "
                    f"'{info.get('name', 'unknown')}'. Please regenerate GT DB."
                )
            occ_list.append(occ_enriched)

        sampled_points = point_list[0].cat(point_list)

        if use_img_branch and len(sampled_gt_bboxes_2d) > 0:
            sampled_gt_bboxes_2d = np.concatenate(sampled_gt_bboxes_2d, axis=0)
            render_records = []
            origin_depths = (gt_bboxes_3d[:, 0] ** 2 +
                             gt_bboxes_3d[:, 1] ** 2)
            sampled_depths = (sampled_gt_bboxes_3d[:, 0] ** 2 +
                              sampled_gt_bboxes_3d[:, 1] ** 2)
            for instance_idx, boxes in enumerate(gt_bboxes_2d_multiview):
                for box in boxes:
                    render_records.append((
                        origin_depths[instance_idx], False, instance_idx,
                        box, None))
            for instance_idx, (info, boxes) in enumerate(zip(
                    sampled, sampled_gt_bboxes_2d_multiview)):
                patch_paths = info.get('image_patch_paths')
                if patch_paths is None:
                    patch_paths = [info['path'] + '.png']
                if len(patch_paths) != len(boxes):
                    raise ValueError(
                        "image_patch_paths and box2d_camera_multiview must "
                        f"have equal lengths for {info.get('path')}: "
                        f"{len(patch_paths)} != {len(boxes)}")
                for box, patch_path in zip(boxes, patch_paths):
                    render_records.append((
                        sampled_depths[instance_idx], True, instance_idx,
                        box, patch_path))

            # Restore/paste every visible camera view from far to near.
            render_records.sort(key=lambda item: item[0], reverse=True)
            for _, is_sampled, instance_idx, box2d_camera, patch_path in render_records:
                bbox_2d = np.asarray(box2d_camera[:4], dtype=np.float32)
                camera_idx = int(box2d_camera[4])

                if camera_idx >= self.img_num:
                    continue

                x1, y1, x2, y2 = [int(ii) for ii in bbox_2d]
                img_h, img_w = img[camera_idx].shape[:2]

                if not is_sampled:
                    # Only restore original areas in the current image to avoid shallow copy contamination
                    x1c = max(x1, 0)
                    y1c = max(y1, 0)
                    x2c = min(x2, img_w - 1)
                    y2c = min(y2, img_h - 1)
                    if x2c >= x1c and y2c >= y1c:
                        # img[camera_idx][y1c:y2c + 1, x1c:x2c + 1] = origin_img[camera_idx][y1c:y2c + 1, x1c:x2c + 1]
                        img[camera_idx][y1c:y2c + 1, x1c:x2c + 1] = self.mixup * origin_img[camera_idx][y1c:y2c + 1, x1c:x2c + 1] + \
                                                                    (1 - self.mixup) * img[camera_idx][y1c:y2c + 1, x1c:x2c + 1]
                else:
                    img_file_path = (os.path.join(self.dataset_root, patch_path)
                                     if self.dataset_root else patch_path)
                    patch_results = dict(img_prefix=None, img_info=dict(filename=img_file_path))
                    s_patch = self.img_loader(patch_results)['img']

                    img[camera_idx] = self.paste_obj(img[camera_idx], s_patch,
                                                     bbox_2d=bbox_2d.astype(np.int32))
        else:
            sampled_gt_bboxes_2d = np.empty((0, 5), dtype=np.float32)
            sampled_gt_bboxes_2d_multiview = []

        gt_labels = np.array([self.cat2label[s['name']] for s in sampled], dtype=np.int64)
        ret = dict(
            img=img if use_img_branch else None,
            gt_labels=gt_labels,
            gt_labels_3d=copy.deepcopy(gt_labels),
            gt_bboxes_3d=sampled_gt_bboxes_3d,
            gt_bboxes_2d=sampled_gt_bboxes_2d,
            gt_bboxes_2d_multiview=sampled_gt_bboxes_2d_multiview,
            points=sampled_points,
            group_ids=np.arange(
                gt_bboxes_3d.shape[0],
                gt_bboxes_3d.shape[0] + len(sampled)),
            occ=occ_list)

        return ret

    def paste_obj(self, img, obj_img, bbox_2d):
        """
        Paste the object image patch onto the background image.

        Args:
            img: Background image (H, W, 3)
            obj_img: Object image patch
            bbox_2d: (x1, y1, x2, y2) bounding box

        Returns:
            Modified image with object pasted
        """
        mixup = self.mixup
        margin_ratio = 0.05
        blending_type = self.blending_type

        x1, y1, x2, y2 = [int(v) for v in bbox_2d]
        # the bbox might exceed the img size because the img is different
        img_h, img_w = img.shape[:2]
        w = np.maximum(min(x2, img_w - 1) - x1 + 1, 1)
        h = np.maximum(min(y2, img_h - 1) - y1 + 1, 1)
        obj_img = obj_img[:h, :w]

        # Create object mask with margin
        obj_mask = np.zeros((h, w), dtype=np.float32)
        margin_h = int(margin_ratio * h)
        margin_w = int(margin_ratio * w)
        # Ensure at least 1 pixel of margin-free region
        margin_h = min(margin_h, max(0, (h - 1) // 2))
        margin_w = min(margin_w, max(0, (w - 1) // 2))
        if margin_h < h - margin_h and margin_w < w - margin_w:
            obj_mask[margin_h:h - margin_h, margin_w:w - margin_w] = 1.0

        # Choose a blend option
        if not blending_type:
            blending_op = 'none'
        else:
            if isinstance(blending_type, list):
                blending_choice = np.random.randint(len(blending_type))
                blending_op = blending_type[blending_choice]
            else:
                blending_op = blending_type

        if blending_op and 'poisson' in blending_op:
            # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
            # cv2.MIXED_CLONE mixed the texture, thus is not used.
            if blending_op == 'poisson':
                mode = np.random.choice([1, 3], 1)[0]
            elif blending_op == 'poisson_normal':
                mode = cv2.NORMAL_CLONE
            elif blending_op == 'poisson_transfer':
                mode = cv2.MONOCHROME_TRANSFER
            else:
                raise NotImplementedError(f"Unsupported blending_type: {blending_op}")

            center = (int(x1 + w / 2), int(y1 + h / 2))
            img = cv2.seamlessClone(obj_img, img, (obj_mask * 255).astype(np.uint8), center, mode)
        else:
            if blending_op == 'gaussian':
                obj_mask = cv2.GaussianBlur(obj_mask.astype(np.float32), (5, 5), 2)
            elif blending_op == 'box':
                obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))

            bg = img[y1:y1 + h, x1:x1 + w].astype(np.float32)
            fg = obj_img.astype(np.float32)

            out = bg * (1.0 - obj_mask[..., None] * mixup) + fg * (obj_mask[..., None] * mixup)
            img[y1:y1 + h, x1:x1 + w] = np.clip(out, 0, 255).astype(np.uint8)

        return img

    def sample_class_vis_aware(self, name, num, gt_bboxes_3d, gt_bboxes_2d):

        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)

        num_gt = gt_bboxes_3d.shape[0]
        num_sampled = len(sampled)
        if num_sampled == 0:
            return []
        # avoid collision in BEV first
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes[:, 0:2], sp_boxes[:, 3:5], sp_boxes[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        # Then avoid collision in 2D space
        if self.check_2D_collision:
            if gt_bboxes_2d is None:
                raise ValueError(
                    "check_2D_collision=True requires gt_bboxes_2d to be provided."
                )

            sp_boxes_2d = [self._db_multiview_boxes(i) for i in sampled]
            all_instance_boxes = list(gt_bboxes_2d) + sp_boxes_2d
            flat_boxes = []
            view_owner = []
            for owner, instance_boxes in enumerate(all_instance_boxes):
                for box in instance_boxes:
                    flat_boxes.append(box)
                    view_owner.append(owner)

            if flat_boxes:
                total_bbox_2d = np.asarray(flat_boxes, dtype=np.float32).reshape(-1, 5)
                total_cam_idx = total_bbox_2d[:, 4].astype(np.int64)
                view_owner = np.asarray(view_owner, dtype=np.int64)

                # random select a collision threshold
                if isinstance(self.collision_thr, (float, int)):
                    collision_thr = float(self.collision_thr)
                elif isinstance(self.collision_thr, list):
                    collision_thr = float(np.random.choice(self.collision_thr))
                elif isinstance(self.collision_thr, dict):
                    mode = self.collision_thr.get('mode', 'value')
                    if mode == 'value':
                        collision_thr = float(np.random.choice(
                            self.collision_thr['thr_range']))
                    elif mode == 'range':
                        collision_thr = np.random.uniform(
                            self.collision_thr['thr_range'][0],
                            self.collision_thr['thr_range'][1]
                        )
                    else:
                        raise ValueError(f"Unknown collision_thr mode: {mode}")
                else:
                    raise TypeError(
                        f"Unsupported type for collision_thr: {type(self.collision_thr)}")

                # Calculate view-level collisions.
                if collision_thr == 0:
                    x1y1 = total_bbox_2d[:, :2]
                    x2y2 = total_bbox_2d[:, 2:4]
                    x1y2 = np.stack(
                        [total_bbox_2d[:, 0], total_bbox_2d[:, 3]], axis=-1)
                    x2y1 = np.stack(
                        [total_bbox_2d[:, 2], total_bbox_2d[:, 1]], axis=-1)
                    total_2d = np.stack(
                        [x1y1, x2y1, x2y2, x1y2], axis=1)
                    view_coll_mat = data_augment_utils.box_collision_test(
                        total_2d, total_2d)
                else:
                    # Use IOF in both directions to protect foregrounds of
                    # different sizes.
                    boxes_xyxy = total_bbox_2d[:, :4]
                    overlaps = box_np_ops.iou_jit(
                        boxes_xyxy, boxes_xyxy, 'iof')
                    view_coll_mat = np.logical_or(
                        overlaps > collision_thr,
                        overlaps.T > collision_thr)

                # Cross-camera boxes never collide. Reduce remaining
                # view-level collisions to their owning 3D instances.
                same_cam_mask = total_cam_idx[:, None] == total_cam_idx[None, :]
                view_coll_mat = np.logical_and(view_coll_mat, same_cam_mask)
                instance_coll_mat = np.zeros_like(coll_mat, dtype=bool)
                rows, cols = np.nonzero(view_coll_mat)
                instance_coll_mat[view_owner[rows], view_owner[cols]] = True
                np.fill_diagonal(instance_coll_mat, False)

                coll_mat = np.logical_or(coll_mat, instance_coll_mat)

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])

        return valid_samples
