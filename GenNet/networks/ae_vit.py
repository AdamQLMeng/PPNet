import numpy as np
import torch
import torch.nn as nn
from functools import partial

from utils import misc
from networks.base import PatchMerging, PatchUpsampling, token2feature, feature2token
from networks.vanilla_swin import BasicLayer
from networks.vit import Block


class AEViT(nn.Module):
    def __init__(self, img_channels, out_channels, img_resolution=256, dim=192):
        super().__init__()
        res = 28

        self.conv_first = nn.Sequential(
                                nn.Conv2d(in_channels=img_channels, out_channels=dim,
                                          kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.LeakyReLU())
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=dim, out_channels=dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU()))

        # from 64 -> 16 -> 64
        depth = 3
        num_heads = 3
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.vit_blocks = nn.Sequential(*[
            Block(dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                  drop_ratio=0, attn_drop_ratio=0, drop_path_ratio=dpr[i],
                  norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
            for i in range(depth)
        ])

        self.dec_conv = nn.ModuleList()
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=dim,
                                       out_channels=dim,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU()))

        self.conv_final = nn.Conv2d(in_channels=dim, out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_first(x)  # input size
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x = block(x)

        _, _, H, W = x.shape
        x = feature2token(x)
        x = self.vit_blocks(x)
        x = token2feature(x, H, W).contiguous()

        for i, block in enumerate(self.dec_conv):
            x = block(x)

        x = self.conv_final(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch = 8
    res = 224
    model = AEViT(img_channels=1, out_channels=1, img_resolution=res).to(device)
    img = torch.randn(batch, 1, res, res).to(device)
    model.eval()

    with torch.no_grad():
        misc.print_module_summary(model, img)
