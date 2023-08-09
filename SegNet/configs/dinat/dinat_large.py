_base_ = [
    '../_base_/models/dinat.py', '../_base_/datasets/planning_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_8k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='DiNAT',
        embed_dim=192,
        mlp_ratio=2.0,
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        kernel_size=7,
        drop_path_rate=0.3,
        dilations=[[1, 20, 1], [1, 5, 1, 10], [1, 2, 1, 3, 1, 4, 1, 5, 1, 2, 1, 3, 1, 4, 1, 5, 1, 5], [1, 2, 1, 2, 1]],
        pretrained='/home/long/pre-train/dinat/dinat_large_in22k_224.pth',
        # 'https://shi-labs.com/projects/dinat/checkpoints/imagenet22k/dinat_large_in22k_224.pth'
    ),
    decode_head=dict(
        in_channels=1536,
        num_convs=4,
        up_scale=2,
        num_classes=2),
)

# optimizer
optimizer = dict(lr=0.02, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=8)

# No mixed precision with float16 in DiNAT-L
#fp16 = None
#optimizer_config = dict(
#    type="Fp16OptimizerHook",
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,
#)
