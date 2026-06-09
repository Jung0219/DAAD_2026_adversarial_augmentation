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

model = dict(
    type='AdvMVXFasterRCNN',
    adv_eps=0.031,
    adv_alpha=0.00627,
    adv_steps=5,
    adv_lambda=1.0
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(ann_file='data/nuscenes/nuscenes_mini_infos_train.pkl'),
    val=dict(ann_file='data/nuscenes/nuscenes_mini_infos_val.pkl'),
    test=dict(ann_file='data/nuscenes/nuscenes_mini_infos_val.pkl')
)

runner = dict(max_epochs=5)
evaluation = dict(interval=5)
lr_config = dict(step=[4], warmup_iters=100)
