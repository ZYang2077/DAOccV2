import argparse
from os import path as osp
import pickle
from data_converter import nuscenes_converter as nuscenes_converter
from data_converter.create_gt_database import create_groundtruth_database
from data_converter import kitti_converter as kitti
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}

classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def add_ann_adj_info(extra_tag):
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    for set in ['train', 'val']:
        dataset = pickle.load(open('%s/%s_infos_%s.pkl' % (dataroot, extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 100 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            # ann_infos = list()  # ['ann_infos'] is not used in this repository
            # for ann in sample['anns']:
            #     ann_info = nuscenes.get('sample_annotation', ann)
            #     velocity = nuscenes.box_velocity(ann_info['token'])
            #     if np.any(np.isnan(velocity)):
            #         velocity = np.zeros(3)
            #     ann_info['velocity'] = velocity
            #     ann_infos.append(ann_info)
            # dataset['infos'][id]['ann_infos'] = ann_infos
            # dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']

            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['scene_name'] = scene['name']
            # dataset['infos'][id]['occ_path'] = \
            #     './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
            # TODO: [yz] support other occ dataset, add dataroot in occ_path
            dataset['infos'][id]['occ3d'] = {
                'occ_path': './data/nuscenes/gts/{}/{}'.format(scene['name'], info['token'])
            }
        with open('%s/%s_infos_%s.pkl' % (dataroot, extra_tag, set), 'wb') as fid:
            pickle.dump(dataset, fid)


def nuscenes_data_prep(
    root_path,
    info_prefix,
    version,
    out_dir,
    max_sweeps=10,
    load_augmented=None,
    with_occ=False,
    occ_data_type='occ3d',
):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    if load_augmented is None:
        # otherwise, infos must have been created, we just skip.
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version, max_sweeps=max_sweeps
        )

        # if version == "v1.0-test":
        #     info_test_path = osp.join(root_path, f"{info_prefix}_infos_test.pkl")
        #     nuscenes_converter.export_2d_annotation(root_path, info_test_path, version=version)
        #     return

        # info_train_path = osp.join(root_path, f"{info_prefix}_infos_train.pkl")
        # info_val_path = osp.join(root_path, f"{info_prefix}_infos_val.pkl")
        # nuscenes_converter.export_2d_annotation(root_path, info_train_path, version=version)
        # nuscenes_converter.export_2d_annotation(root_path, info_val_path, version=version)

    add_ann_adj_info(info_prefix)
    # TODO: [yz] 3D box from ann_infos is in ego-vehicle coordinate system, and its wlh has been converted to lwh.

    create_groundtruth_database(
        "NuScenesDatasetOccupancy" if with_occ else "NuScenesDataset",
        root_path,
        info_prefix,
        f"{out_dir}/{info_prefix}_infos_train.pkl",
        load_augmented=load_augmented,
        with_occ=with_occ,
        occ_data_type=occ_data_type,
    )

def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=5,
                    with_occ=False):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    from data_converter import waymo_converter as waymo

    splits = ['training', 'validation', 'testing']
    for i, split in enumerate(splits):
        load_dir = osp.join(root_path, 'waymo_format', split)
        if split == 'validation':
            save_dir = osp.join(out_dir, 'kitti_format', 'training')
        else:
            save_dir = osp.join(out_dir, 'kitti_format', split)
        converter = waymo.Waymo2KITTI(
            load_dir,
            save_dir,
            prefix=str(i),
            workers=workers,
            test_mode=(split == 'test'))
        converter.convert()
    # Generate waymo infos
    out_dir = osp.join(out_dir, 'kitti_format')
    kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)
    create_groundtruth_database(
        'WaymoDatasetOccupancy' if with_occ else 'WaymoDataset',
        out_dir,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        with_occ=with_occ
    )


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", metavar="kitti", help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    default="./data/kitti",
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    default="v1.0",
    required=False,
    help="specify the dataset version, no need for kitti",
)
parser.add_argument(
    "--max-sweeps",
    type=int,
    default=10,
    required=False,
    help="specify sweeps of lidar per example",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="./data/kitti",
    required=False,
    help="name of info pkl",
)
parser.add_argument("--extra-tag", type=str, default="kitti")
parser.add_argument("--painted", default=False, action="store_true")
parser.add_argument("--virtual", default=False, action="store_true")

# occ related
parser.add_argument(
    "--with-occ",
    default=False,
    action="store_true",
    help="enable occupancy-aware GT database generation",
)
parser.add_argument(
    "--occ-data-type",
    type=str,
    default="occ3d",
    choices=["occ3d", "surround_occ", "open_occ"],
    help="occupancy data type used by GT database generation",
)

parser.add_argument(
    "--workers", type=int, default=4, help="number of threads to be used"
)
args = parser.parse_args()

if __name__ == "__main__":
    load_augmented = None
    if args.virtual:
        if args.painted:
            load_augmented = "mvp"
        else:
            load_augmented = "pointpainting"

    if args.dataset == "nuscenes" and args.version != "v1.0-mini":
        train_version = f"{args.version}-trainval"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
            with_occ=args.with_occ,
            occ_data_type=args.occ_data_type,
        )
        # test_version = f"{args.version}-test"
        # nuscenes_data_prep(
        #     root_path=args.root_path,
        #     info_prefix=args.extra_tag,
        #     version=test_version,
        #     out_dir=args.out_dir,
        #     max_sweeps=args.max_sweeps,
        #     load_augmented=load_augmented,
        # )
    elif args.dataset == "nuscenes" and args.version == "v1.0-mini":
        train_version = f"{args.version}"
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            load_augmented=load_augmented,
            with_occ=args.with_occ,
            occ_data_type=args.occ_data_type,
        )
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps,
            with_occ=args.with_occ,
        )
