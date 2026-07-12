import argparse
import time
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.parallel import DataContainer

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_fusion_model
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet benchmark vtransform + fuser module")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--samples", type=int, default=2000, help="samples to benchmark")
    parser.add_argument("--log-interval", default=50, help="interval of logging")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    return args

# Utility function: Unpack MMDetection DataContainer
def unwrap_data(data):
    if isinstance(data, DataContainer):
        return data.data[0] if len(data.data) > 0 else data.data
    return data

def main():
    args = parse_args()

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # Build data loader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # Build model
    cfg.model.train_cfg = None
    model = build_fusion_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if args.fp16:
        wrap_fp16_model(model)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # ===================== Extract all speed measurement modules =====================
    # camera branch
    cam_encoder = model.module.encoders.camera
    cam_backbone = cam_encoder.backbone
    cam_neck = cam_encoder.neck
    vtransform = cam_encoder.vtransform
    
    # radar branch
    lidar_encoder = model.module.encoders.lidar
    lidar_backbone = lidar_encoder.backbone
    voxelize_func = model.module.voxelize
    
    # Fusion module fuser (new speed measurement target)
    fuser = model.module.fuser
    # ===========================================================

    # All set to evaluation mode
    cam_backbone.eval()
    cam_neck.eval()
    vtransform.eval()
    lidar_backbone.eval()
    fuser.eval()

    # Preheat + timing variable
    num_warmup = 5
    pure_vtransform_time = 0  # vtransform total time taken
    pure_fuser_time = 0       # fuser total time spent

    # Cycle speed test
    for i, data in enumerate(data_loader):
        # 1. Unpack all data
        img = unwrap_data(data['img']).cuda()
        points = [p.cuda() for p in unwrap_data(data['points'])]
        camera2ego = unwrap_data(data['camera2ego']).cuda()
        lidar2ego = unwrap_data(data['lidar2ego']).cuda()
        lidar2camera = unwrap_data(data['lidar2camera']).cuda()
        lidar2image = unwrap_data(data['lidar2image']).cuda()
        camera_intrinsics = unwrap_data(data['camera_intrinsics']).cuda()
        camera2lidar = unwrap_data(data['camera2lidar']).cuda()
        img_aug_matrix = unwrap_data(data['img_aug_matrix']).cuda()
        lidar_aug_matrix = unwrap_data(data['lidar_aug_matrix']).cuda()
        img_metas = unwrap_data(data['metas'])

        # 2. Radar voxelization (pre-calculation, time-consuming not included)
        with torch.no_grad():
            voxel_outputs = voxelize_func(points)
            if len(voxel_outputs) == 4:
                feats, coords, sizes, voxel_coords_list = voxel_outputs
            else:
                feats, coords, sizes = voxel_outputs
                voxel_coords_list = None

        # 3. Camera feature extraction (pre-calculation, time-consuming not included)
        with torch.no_grad():
            B, N, C, H, W = img.size()
            x = img.view(B * N, C, H, W)
            x = cam_backbone(x)
            x = cam_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            BN, C, H, W = x.size()
            x = x.view(B, int(BN / B), C, H, W)

        # 4. ===================== Speed ​​test 1: vtransform module =====================
        torch.cuda.synchronize()
        start_v = time.perf_counter()
        with torch.no_grad():
            cam_bev_feat = vtransform(
                x, points, camera2ego, lidar2ego, lidar2camera, lidar2image,
                camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix,
                img_metas, voxel_coords_list=voxel_coords_list
            )
        torch.cuda.synchronize()
        elapsed_v = time.perf_counter() - start_v
        # ====================================================================

        # 5. Radar BEV feature extraction (pre-calculation, time-consuming not included)
        with torch.no_grad():
            batch_size = coords[-1, 0] + 1
            lidar_bev_feat = lidar_backbone(feats, coords, batch_size, sizes=sizes)

        # 6. ===================== Speed ​​test 2: fuser fusion module =====================
        torch.cuda.synchronize()
        start_f = time.perf_counter()
        with torch.no_grad():
            # Strictly follow the model logic: input [camera features, radar features]
            fused_feat = fuser([cam_bev_feat, lidar_bev_feat])
        torch.cuda.synchronize()
        elapsed_f = time.perf_counter() - start_f
        # ====================================================================

        # Time consuming statistics after warm-up
        if i >= num_warmup:
            pure_vtransform_time += elapsed_v
            pure_fuser_time += elapsed_f

            # Scheduled printing
            if (i + 1) % args.log_interval == 0:
                total_infer = i + 1 - num_warmup
                avg_v_ms = (pure_vtransform_time / total_infer) * 1000
                avg_f_ms = (pure_fuser_time / total_infer) * 1000
                print(
                    f"Done sample [{i + 1:<3}/ {args.samples}], "
                    f"vtransform: {avg_v_ms:.3f} ms | fuser: {avg_f_ms:.3f} ms"
                )

        # final result
        if (i + 1) == args.samples:
            total_infer = i + 1 - num_warmup
            avg_v_ms = (pure_vtransform_time / total_infer) * 1000
            avg_f_ms = (pure_fuser_time / total_infer) * 1000
            print(f"\n==== 最终测速结果 ====")
            print(f"vtransform 平均耗时: {avg_v_ms:.3f} ms")
            print(f"fuser 融合平均耗时: {avg_f_ms:.3f} ms")
            print(f"总测试样本: {args.samples} | 预热样本: {num_warmup}")
            break

if __name__ == "__main__":
    main()
