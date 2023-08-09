# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize, Upsample
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@HEADS.register_module()
class UPerPUPHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,
                 num_convs=(2, 3, 4, 5),
                 up_scale=2,
                 pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerPUPHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i, in_channels in enumerate(self.in_channels):
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = nn.ModuleList()
            for _ in range(num_convs[i]):
                fpn_conv.append(
                    nn.Sequential(
                        ConvModule(
                            in_channels=self.channels,
                            out_channels=self.channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg),
                        Upsample(
                            scale_factor=up_scale,
                            mode='bilinear',
                            align_corners=self.align_corners)))
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        self.lateral_convs = self.lateral_convs[:-1]

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)
        # print('input', [xx.shape for xx in inputs])
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))
        used_backbone_levels = len(laterals)
        # print('lateral', [xx.shape for xx in laterals])
        # build top-down path
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)
        # print('lateral', [xx.shape for xx in laterals])
        # build outputs
        fpn_outs = []
        for i in range(used_backbone_levels):
            xx = laterals[i]
            for conv in self.fpn_convs[i]:
                xx = conv(xx)
            fpn_outs.append(xx)
        # print('out', [xx.shape for xx in fpn_outs])
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.cls_seg(output)
        return output
