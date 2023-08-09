# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DiNAT',
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        kernel_size=7,
        dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        in_patch_size=4,
        frozen_stages=-1,
    ),
    decode_head=dict(
        type='SETRUPHead',
        norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
        num_convs=2,
        up_scale=4,
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
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
