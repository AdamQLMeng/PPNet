_base_ = [
    '../_base_/models/dinat.py', '../_base_/datasets/planning_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_8k.py'
]
model = dict(
    backbone=dict(
        type='DiNAT',
        embed_dim=128,
        mlp_ratio=2.0,
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.5,
        kernel_size=7,
        layer_scale=1e-5,
        dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
        pretrained='/home/long/pre-train/dinat/dinat_base_in1k_224.pth',
        # 'https://shi-labs.com/projects/dinat/checkpoints/imagenet1k/dinat_base_in1k_224.pth'
    ),
    decode_head=dict(
        in_channels=1024,
        num_convs=4,
        up_scale=2,
        num_classes=2),
)

# optimizer
optimizer = dict(lr=0.02, weight_decay=0.0,
                 paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)})
                 )

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

# Mixed precision
fp16 = None
optimizer_config = dict(
    type="Fp16OptimizerHook",
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
)
