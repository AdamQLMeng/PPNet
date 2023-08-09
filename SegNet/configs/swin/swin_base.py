_base_ = [
    '../_base_/models/swin.py', '../_base_/datasets/planning_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_8k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        drop_path_rate=0.3,
        patch_norm=True,
        pretrained='/home/long/pre-train/swin/swin_base_patch4_window7_224.pth'
    ),
    decode_head=dict(
        type='SETRUPHead',
        norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
        num_convs=4,
        up_scale=2,
        kernel_size=3,
        init_cfg=[
            dict(type='Constant', val=1.0, bias=0, layer='LayerNorm'),
            dict(
                type='Normal',
                std=0.01,
                override=dict(name='conv_seg'))
        ],
        in_channels=1024,
        channels=512,
        in_index=-1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

# optimizer
optimizer = dict(lr=0.08, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8)
