_base_ = ['../../../../configs/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py']

custom_imports = dict(
    imports=['projects.adv_aug.plugins'],
    allow_failed_imports=False
)

load_from = 'projects/adv_aug/checkpoints/centerpoint/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20200925_230905-358fbe3b.pth'
