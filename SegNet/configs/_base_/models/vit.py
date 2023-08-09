# model settings
norm_layer = dict(type='LN', eps=1e-6, requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=12,
        num_heads=12,
        drop_rate=0.1,
        norm_cfg=norm_layer,
        with_cls_token=False,
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
        in_channels=768,
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
