_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn.py',
    # '../_base_/datasets/coco_instance.py',
    '../_base_/datasets/coco_detection.py',  # use detection
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=True,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    neck=dict(in_channels=[96, 192, 384, 768]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageAroundFile', num_channels=3),
    dict(type='LoadAnnotations', with_bbox=True),  # remove mask
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(852, 852), (906, 906), (956, 956), (1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(852, 852), (906, 906), (956, 956), (1024, 1024)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(852, 852),
                      allow_negative_crop=True)
             ]
         ]),
    # dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),  # remove mask
]
test_pipeline = [
    dict(type='LoadImageAroundFile', num_channels=3, is_testing=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(852, 852),
        # flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(train=dict(pipeline=train_pipeline))
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        classes=('js', 'jc', 'xs'),
        ann_file='train.json',
        img_prefix='/data/common/OCT_data/img',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=('js', 'jc', 'xs'),
        ann_file='test.json',
        img_prefix='/data/common/OCT_data/img',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=('js', 'jc', 'xs'),
        ann_file='test.json',
        img_prefix='/data/common/OCT_data/img',
        pipeline=test_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# lr_config = dict(step=[15, 35])
lr_config = dict(step=[15, 39])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=40)

# do not use mmdet version fp16
# fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)

# load_from = "mask_rcnn_swin_tiny_patch4_window7.pth"
# load_from = "mask_rcnn_swin_small_patch4_window7.pth"
load_from = "/data/common/OCT_data/grid0/epoch_1.pth"

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook')
    ])
log_level = 'INFO'
work_dir = "logs"
