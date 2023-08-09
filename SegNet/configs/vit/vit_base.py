_base_ = [
    '../_base_/models/setr.py', '../_base_/datasets/planning_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_8k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        drop_rate=0.,
        pretrained='/home/long/pre-train/vit/vit_base_patch16_224.pth',
        # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
        # 'https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-weights/vit_base_p16_224-4e355ebd.pth'
    ),
    decode_head=dict(
        in_channels=768,
        num_convs=4,
        up_scale=2,
        num_classes=2),
)

# optimizer
optimizer = dict(lr=0.02, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4)
