# Training and Evaluation

## Training

Download the ResNet-50 image-backbone initialization. A processed `htc_r50_backbone.pth` is available [here](https://drive.google.com/file/d/19S91tPjfM2laHKipL7QDs5-6m9pQxQGz/view?usp=drive_link).

Train with 4 GPUs:

```shell
bash tools/dist_train.sh PATH_TO_CONFIG 4 \
  --run-dir CHECKPOINT_SAVE_DIR \
  --model.encoders.camera.backbone.init_cfg.checkpoint PATH_TO_PRETRAIN
```

Train with 1 GPU:

```shell
python tools/train.py PATH_TO_CONFIG \
  --run-dir CHECKPOINT_SAVE_DIR \
  --model.encoders.camera.backbone.init_cfg.checkpoint PATH_TO_PRETRAIN
```

`PATH_TO_PRETRAIN` may be the processed ResNet-50 checkpoint described above. Training writes the resolved configuration, logs, and checkpoints to `CHECKPOINT_SAVE_DIR`.

## Evaluation

Evaluate with 4 GPUs:

```shell
bash tools/dist_test.sh PATH_TO_CONFIG PATH_TO_WEIGHT 4
```

Evaluate with 1 GPU:

```shell
python tools/test.py PATH_TO_CONFIG PATH_TO_WEIGHT
```

The evaluation protocol and masks are controlled by the selected configuration. In particular, use the `wo_mask` Occ3D-nuScenes configuration when reporting RayIoU without camera masks by adding:
```yaml
evaluation:
  occ_metric: Ray-IoU
```

## Inference benchmark

Measure end-to-end inference throughput on one GPU with:

```shell
python tools/benchmark.py PATH_TO_CONFIG \
  --checkpoint PATH_TO_WEIGHT \
  --samples 50
```
