import sys
import torch
import mmcv
from mmcv import Config
from mmdet3d.models import build_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet3d.datasets import build_dataset as build_dataset3d
from mmdet3d.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

def main():
    cfg = Config.fromfile("projects/adv_aug/configs/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py")
    cfg.data.test.test_mode = True
    
    dataset = build_dataset3d(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)
    
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, "projects/adv_aug/checkpoints/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.pth", map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    print(f"Dataset length: {len(dataset)}")
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
        boxes = result[0]['pts_bbox']['boxes_3d']
        scores = result[0]['pts_bbox']['scores_3d']
        print(f"Sample {i}:")
        print(f"  Num boxes: {len(boxes)}")
        if len(scores) > 0:
            print(f"  Max score: {scores.max().item()}")
        else:
            print(f"  Max score: N/A")
            
        if i >= 1:
            break

if __name__ == "__main__":
    main()
