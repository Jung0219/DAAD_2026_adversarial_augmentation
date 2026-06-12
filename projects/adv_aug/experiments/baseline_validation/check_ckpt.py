import torch

ckpt = torch.load("projects/adv_aug/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.pth", map_location="cpu")
state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

for k, v in state_dict.items():
    if "voxel_encoder" in k or "pfn_layers.0.linear.weight" in k:
        print(f"{k}: {v.shape}")

print("--------------------")
print(ckpt.get('meta', {}).get('CLASSES', "No CLASSES in meta"))
