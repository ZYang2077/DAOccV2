# Dataset Preparation

DAOccV2 supports Occ3D-nuScenes, SurroundOcc, and Occ3D-Waymo.

## nuScenes

### 1. Raw nuScenes data

Download the nuScenes data from the [nuScenes website](https://www.nuscenes.org/download) and arrange it as follows:

```text
DAOccV2/
└── data/
    └── nuscenes/
        ├── maps/
        ├── samples/
        ├── sweeps/
        └── v1.0-trainval/
```

### 2. Occupancy annotations

For **Occ3D-nuScenes**, download `gts.tar.gz` from [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D), extract it, and place the labels under:

```text
data/nuscenes/gts/<scene-name>/<sample-token>/labels.npz
```

Each `labels.npz` is expected to contain `semantics`, `mask_lidar`, and `mask_camera`.

For **SurroundOcc**, download the 200 x 200 x 16 occupancy labels from [SurroundOcc](https://github.com/weiyithu/SurroundOcc) and place them under:

```text
data/nuscenes_occ/samples/<lidar-file-name>.npy
```

### 3. Generate dataset information and the GT database

For Occ3D-nuScenes, run:

```shell
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --with-occ \
  --occ-data-type occ3d
```

For SurroundOcc, run the same preprocessing with the matching occupancy type:

```shell
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --with-occ \
  --occ-data-type surround_occ
```

The main outputs are:

```text
data/nuscenes/
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── nuscenes_dbinfos_train.pkl
└── nuscenes_gt_database/
    ├── <sample>_<class>_<index>.bin
    ├── <sample>_<class>_<index>.bin.png
    └── <sample>_<class>_<index>.bin.cam<camera-index>.png
```

The Occ3D and SurroundOcc commands write the same database filenames, while their entries contain different enriched keys (`occ3d_enriched` and `surround_occ_enriched`). Generate the database for the target benchmark before training and do not reuse an Occ3D-enriched database for SurroundOcc, or vice versa.

### 4. Filter instances by visibility

`tools/db_infos_filter.py` currently uses a configuration block at the top of the file rather than command-line arguments. Set it as follows for Occ3D-nuScenes:

```python
INPUT_PKL_PATH = "data/nuscenes/nuscenes_dbinfos_train.pkl"
VIS_SCORE_THRESHOLD = 0.3
OCC_DATA_TYPE = "occ3d"
```

For SurroundOcc, change only:

```python
OCC_DATA_TYPE = "surround_occ"
```

Then run:

```shell
python tools/db_infos_filter.py
```

The script keeps instances whose `vis_score >= 0.3` and writes:

```text
data/nuscenes/nuscenes_dbinfos_train_vis_thr_0p3.pkl
```

This is the file referenced by all three nuScenes configurations. As with the unfiltered file, its enriched content must match the target benchmark.

### Reference nuScenes layout

```text
DAOccV2/
└── data/
    ├── nuscenes/
    │   ├── gts/
    │   ├── maps/
    │   ├── samples/
    │   ├── sweeps/
    │   ├── v1.0-trainval/
    │   ├── nuscenes_infos_train.pkl
    │   ├── nuscenes_infos_val.pkl
    │   ├── nuscenes_dbinfos_train.pkl
    │   ├── nuscenes_dbinfos_train_vis_thr_0p3.pkl
    │   └── nuscenes_gt_database/
    └── nuscenes_occ/
        └── samples/
```

## Waymo

### 1. Raw Waymo data and Occ3D labels

Download the Waymo Open Dataset TFRecords and organize the three splits as expected by `waymo_data_prep`:

```text
data/waymo/waymo_format/
├── training/*.tfrecord
├── validation/*.tfrecord
└── testing/*.tfrecord
```

Download the Occ3D-Waymo occupancy annotations from [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D), along with `cam_infos.pkl` and `cam_infos_vali.pkl`. The loader expects the expanded `voxel04` labels in this form:

```text
data/waymo/occ3d/voxel04/
├── training/<sequence>/<frame>_04.npz
└── validation/<sequence>/<frame>_04.npz
```

Each archive is expected to contain `voxel_label`, `infov`, `origin_voxel_state`, and `final_voxel_state`.

The current Waymo loader and GT-database generator also require these [metadata files](https://drive.google.com/drive/folders/18BVuF_RYJF0NjZpt8SnfzANiakoRMf0o):

```text
data/waymo/kitti_format/ImageSets/train.txt
data/waymo/kitti_format/ImageSets/val.txt
data/waymo/kitti_format/ImageSets/test.txt
```

### 2. Convert Waymo and generate the OA-GTAUG database

Waymo conversion additionally requires the Waymo Open Dataset development kit described in [Installation](install.md). Run:

```shell
python tools/create_data.py waymo \
  --root-path ./data/waymo \
  --out-dir ./data/waymo \
  --extra-tag waymo \
  --workers 128 \
  --with-occ
```

The converter reads `data/waymo/waymo_format`, writes KITTI-format sensor data to `data/waymo/kitti_format`, creates `waymo_infos_*.pkl`, and then generates:

```text
data/waymo/kitti_format/waymo_gt_database/
data/waymo/kitti_format/waymo_dbinfos_train.pkl
```

The GT-database generator currently reads the Waymo occupancy and camera-pose paths shown above directly from `data/waymo`. Keep those paths unchanged while preprocessing, or update the corresponding code and configuration together.

### 3. Filter Waymo instances by visibility

Configure `tools/db_infos_filter.py` as follows:

```python
INPUT_PKL_PATH = "data/waymo/kitti_format/waymo_dbinfos_train.pkl"
VIS_SCORE_THRESHOLD = 0.3
OCC_DATA_TYPE = "occ3d"
```

Run:

```shell
python tools/db_infos_filter.py
```

The automatically generated output filename then matches the released configuration:

```text
data/waymo/kitti_format/waymo_dbinfos_train_vis_thr_0p3.pkl
```

- Note that the cameras with index 2 and index 3 stored in `cam_infos.pkl` are swapped due to a bug. We directly skip data augmentation for cameras 2 and 3 during the data augmentation pipeline.

### Reference Waymo layout

```text
DAOccV2/
└── data/
    └── waymo/
        ├── waymo_format/
        │   ├── training/
        │   ├── validation/
        │   └── testing/
        ├── occ3d/voxel04/
        │   ├── training/
        │   └── validation/
        └── kitti_format/
            ├── ImageSets/
            ├── training/
            ├── testing/
            ├── cam_infos.pkl
            ├── cam_infos_vali.pkl
            ├── waymo_infos_train.pkl
            ├── waymo_infos_val.pkl
            ├── waymo_gt_database/
            └── waymo_dbinfos_train_vis_thr_0p3.pkl
```