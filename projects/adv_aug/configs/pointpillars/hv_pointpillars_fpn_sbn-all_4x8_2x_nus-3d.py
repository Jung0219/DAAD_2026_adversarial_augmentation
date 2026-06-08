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
