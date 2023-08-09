# dataset settings
dataset_type = 'PascalContextDataset'
data_root = '/home/long/dataset/planning224_1_seg_generalization/'
data_root = '/home/long/dataset/data_BITstar_ppnet/'
data_root = '/home/long/dataset/exp_diff_clearance/'
data_root = '/home/long/dataset/exp_diff_data_generation/'
# data_root = '/home/long/dataset/exp_diff_algo/'
# data_root = '/home/long/dataset/exp_diff_data_generation/Data_BITstar_40*250/'
# data_root = '/home/long/dataset/exp_diff_data_generation/Data_InformedRRTstar_40*250/'
# data_root = '/home/long/dataset/exp_diff_data_generation/Data_RRTstar_40*250/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (224, 224)
crop_size = (224, 224)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='map',
        ann_dir='mask_space',
        split='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='map',
        ann_dir='mask_space',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='map',
        ann_dir='mask_space',
        split='ImageSets/Segmentation/test.txt',
        pipeline=test_pipeline))
