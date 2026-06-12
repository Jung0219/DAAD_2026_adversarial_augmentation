_base_ = [
    '../../../../configs/_base_/models/hv_pointpillars_fpn_nus.py',
    '../../../../configs/_base_/datasets/nus-3d.py',
    '../../../../configs/_base_/schedules/schedule_2x.py',
    '../../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.adv_aug.plugins'],
    allow_failed_imports=False
)

load_from = 'projects/adv_aug/checkpoints/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth'
