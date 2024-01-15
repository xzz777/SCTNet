#1.model
checkpoint_teacher = 'pretrain/Teacher_SegFormer_B3_COCO.pth'
checkpoint_backbone = 'pretrain/SCT-B_Pretrain.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_Distill',
    pretrained=None,
    backbone=dict(
        type='SCTNet',
        init_cfg=dict(
            type='Pretrained',
            checkpoint= checkpoint_backbone
        ),
        base_channels=64,
        spp_channels=128),
    decode_head=dict(
        type='SCTHead',
        in_channels=256,
        channels=256,
        dropout_ratio=0.0,
        in_index=0,
        num_classes=171,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='AU_SCTHead',
            in_channels=128,
            channels=128,
            dropout_ratio=0.0,
            in_index=1,
            num_classes=171,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4)),
        dict(
            type='VitGuidanceHead',
            init_cfg=dict(
                type='Pretrained',
                checkpoint= checkpoint_teacher),
            in_channels=256,
            channels=256,
            base_channels=64,
            in_index=2,
            num_classes=171,
            loss_decode=dict(type='AlignmentLoss', loss_weight=[3, 15, 15, 15]))
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

#2. data
dataset_type = 'COCOStuffDataset'
data_root = 'data/coco_stuff10k'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2560, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='COCOStuffDataset',
        data_root='data/coco_stuff10k',
        reduce_zero_label=True,
        img_dir='images/train2014',
        ann_dir='annotations/train2014',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(2560, 640), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(640, 640), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='COCOStuffDataset',
        data_root='data/coco_stuff10k',
        reduce_zero_label=True,
        img_dir='images/test2014',
        ann_dir='annotations/test2014',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2560, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='COCOStuffDataset',
        data_root='data/coco_stuff10k',
        reduce_zero_label=True,
        img_dir='images/test2014',
        ann_dir='annotations/test2014',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2560, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

#optimzer
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            teacher_backbone=dict(lr_mult=0.0),
            teacher_head=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
find_unused_parameters = True
auto_resume = True
seed = 1744907615