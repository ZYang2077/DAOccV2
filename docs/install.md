# Installation

The codebase is built on PyTorch 1.10, MMCV 1.x, and MMDetection 2.x. The following environment matches the versions used by the project.

## 1. Create the environment

```shell
conda create -n daoccv2 python=3.8 -y
conda activate daoccv2
```

## 2. Install PyTorch

Install PyTorch 1.10.2 and torchvision 0.11.3 for CUDA 11.3:

```shell
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

If a different CUDA toolkit is used, select the matching PyTorch build and ensure it remains compatible with the MMCV build installed below.

## 3. Install MMCV and the Python dependencies

```shell
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.20.0
pip install nuscenes-devkit==1.1.11
pip install torchpack==0.3.1
pip install numba==0.48.0
pip install numpy==1.22.0
pip install prettytable==3.11.0
pip install ninja==1.11.1.1
pip install einops==0.8.0
pip install yapf==0.40.0
pip install pillow==8.4.0
pip install Shapely==1.7.1
pip install pycocotools==2.0.7
pip install scikit-image==0.19.3
pip install open3d==0.19.0
```

## 4. Waymo development kit (optional)

This step is required when preprocessing Occ3D-Waymo. The converter currently imports TensorFlow and the Waymo package exposed by this development-kit build:

```shell
pip install waymo-open-dataset-tf-2-6-0==1.4.8
```

## 5. Install DAOccV2

```shell
git clone https://github.com/ZYang2077/DAOccV2.git
cd DAOccV2
python setup.py develop
```
