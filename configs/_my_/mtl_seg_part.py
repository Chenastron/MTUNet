_base_ = [
    '../_base_/datasets/test_mtl.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# custom_imports = dict(imports=['mmseg.models', 'models', 'datasets', 'datasets.pipelines'], allow_failed_imports=False)
custom_imports = dict(imports=['mmseg.models'], allow_failed_imports=False)

norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
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
    decode_head=dict(
        type='mmseg.PSPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.1, 10.0]
        )),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.1, 10.0]
        )),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='slide', crop_size=256, stride=170))
    test_cfg=dict(mode='whole'))

# We fixed the incorrect img_norm_cfg problem in the source code.
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True)
crop_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(
    #     type='PhotoMetricDistortion',
    #     brightness_delta=32,
    #     contrast_range=(0.5, 1.5),
    #     saturation_range=(0.5, 1.5),
    #     hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=crop_size,
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True,
        test_pad_mode=None),
    dict(type='Resize', img_scale=crop_size, keep_ratio=True, interpolation='nearest'),
    # dict(type='Resize', img_scale=crop_size, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, interpolation='nearest'),
            # dict(type='Resize', keep_ratio=True),
            dict(
                type='RandomCenterCropPad',
                ratios=None,
                border=None,
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True,
                test_mode=True,
                test_pad_mode=['logical_or', 31],
                test_pad_add_pix=1),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'border', 'ori_filename'),
                keys=['img'])
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
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        img_dir=data_root + 'img_dir/train',
        ann_dir=data_root + 'ann_dir/train',
        seg_prefix=data_root + 'ann_dir/train',
        pipeline=train_pipeline,
        classes=classes,
        seg_classes=seg_classes),
    # train=dict(
    #     type='RepeatDataset',
    #     times=5,
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file=data_root + 'annotations/instances_train2017.json',
    #         img_prefix=data_root + 'train2017/',
    #         img_dir=data_root + 'img_dir/train',
    #         ann_dir=data_root + 'ann_dir/train',
    #         seg_prefix=data_root + 'ann_dir/train',
    #         pipeline=train_pipeline,
    #         classes=classes,
    #         seg_classes=seg_classes)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_dir=data_root + 'img_dir/val',
        ann_dir=data_root + 'ann_dir/val',
        seg_prefix=data_root + 'ann_dir/val',
        pipeline=test_pipeline,
        classes=classes,
        seg_classes=seg_classes),
    test=dict(
        type='RepeatDataset',  # RepeatDataset for test fps
        times=500,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_val2017.json',
            img_prefix=data_root + 'val2017/',
            img_dir=data_root + 'img_dir/val',
            ann_dir=data_root + 'ann_dir/val',
            seg_prefix=data_root + 'ann_dir/val',
            pipeline=test_pipeline,
            classes=classes)))

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_config = dict(_delete_=True, policy='poly', power=2.0, min_lr=0.00000, by_epoch=False)
runner = dict(max_epochs=200)

log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        # dict(type='WandbLoggerHook'),
        # dict(type='NeptuneLoggerHook',
        #      init_kwargs=dict(project='chenastron/mtl',
        #                       mode='offline'
        #                       ))
    ])
# evaluation = dict(metric='mIoU', save_best='mIoU', pre_eval=True)
evaluation = dict(metric='mIoU', save_best='mIoU')
# workflow = [('train', 1), ('val', 1)]
checkpoint_config = dict(interval=1)
