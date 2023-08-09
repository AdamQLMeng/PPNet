import numpy as np
import torch
import torch.nn as nn

from utils import misc
from networks.base import PatchMerging, PatchUpsampling, token2feature, feature2token
from networks.vanilla_swin import BasicLayer


class AESwin(nn.Module):
    def __init__(self, img_channels, out_channels, img_resolution=256, dim=192):
        super().__init__()
        res = 56

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
        depths = [2, 3, 4, 3, 2]
        ratios = [1/2, 1/2, 1, 2, 2]
        num_heads = [6, 12, 24, 12, 6]
        window_sizes = [7, 14, 14, 14, 7]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.swin = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1/ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.swin.append(
                BasicLayer(dim=dim, depth=depth, num_heads=num_heads[i],
                           window_size=window_sizes[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                           downsample=merge)
            )

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
        for layer in self.swin:
            x, H, W = layer(x, H, W)
        x = token2feature(x, H, W).contiguous()

        for i, block in enumerate(self.dec_conv):
            x = block(x)

        x = self.conv_final(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch = 8
    res = 224
    model = AESwin(img_channels=1, out_channels=1, img_resolution=res).to(device)
    img = torch.randn(batch, 1, res, res).to(device)
    model.eval()

    with torch.no_grad():
        misc.print_module_summary(model, img)
