_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/test_mtl.py',
    '../_base_/schedules/schedule_40k.py', '../_base_/default_runtime.py'
]

# custom_imports = dict(imports=['mmseg.models', 'models', 'datasets', 'datasets.pipelines'], allow_failed_imports=False)
custom_imports = dict(imports=['mmseg.models'], allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='mmseg.UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    neck=dict(in_channels=[64, 128, 256, 512, 1024],
              start_level=0,
              ),
    # backbone=dict(
    #     depth=18,
    #     init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    # neck=dict(in_channels=[64, 128, 256, 512]),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# We fixed the incorrect img_norm_cfg problem in the source code.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'MtlDataset'
data_root = 'data/sirst/'
classes = ('target',)
seg_classes = ('background', 'target')

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    # train=dict(
    #     type=dataset_type,
    #     ann_file=data_root + 'annotations/instances_train2017.json',
    #     img_prefix=data_root + 'train2017/',
    #     img_dir=data_root + 'img_dir/train',
    #     ann_dir=data_root + 'ann_dir/train',
    #     seg_prefix=data_root + 'ann_dir/train',
    #     pipeline=train_pipeline,
    #     classes=classes,
    #     seg_classes=seg_classes),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            img_dir=data_root + 'img_dir/train',
            ann_dir=data_root + 'ann_dir/train',
            seg_prefix=data_root + 'ann_dir/train',
            pipeline=train_pipeline,
            classes=classes,
            seg_classes=seg_classes)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_dir=data_root + 'img_dir/val',
        ann_dir=data_root + 'ann_dir/val',
        seg_prefix=data_root + 'ann_dir/val',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_dir=data_root + 'img_dir/val',
        ann_dir=data_root + 'ann_dir/val',
        seg_prefix=data_root + 'ann_dir/val',
        pipeline=test_pipeline,
        classes=classes))

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_config = dict(_delete_=True, policy='poly', power=2.0, min_lr=0.00000, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=400)
evaluation = dict(interval=400, metric='bbox', save_best='bbox_mAP_50')
auto_scale_lr = dict(base_batch_size=4)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook'),
        dict(type='NeptuneLoggerHook',
             init_kwargs=dict(project='chenastron/mtl',
                              mode='offline'
                              ))
    ])


