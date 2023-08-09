import torch.nn as nn


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = nn.Sequential(
                                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=3, stride=down, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU())
        self.down = down

    def forward(self, x, H, W):
        x = token2feature(x, H, W)
        x = self.conv(x)
        if self.down != 1:
            ratio = 1 / self.down
            H, W = (int(H * ratio), int(W * ratio))
        x = feature2token(x)
        return x, H, W


class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = nn.Sequential(
                                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=3, stride=up, padding=1, output_padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.LeakyReLU())
        self.up = up

    def forward(self, x, H, W):
        x = token2feature(x, H, W)
        x = self.conv(x)
        if self.up != 1:
            H, W = (int(H * self.up), int(W * self.up))
        x = feature2token(x)
        return x, H, W


def token2feature(x, H, W):
    B, N, C = x.shape
    x = x.permute(0, 2, 1).reshape(B, C, H, W)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x
