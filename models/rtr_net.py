import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class RTRNet(nn.Module):
    """Lightweight U-Net style RTR-Net.

    Input: RGB image and response map concatenated as a 4-channel tensor.
    Output: rectified mask probability and auxiliary skeleton probability.
    """

    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        c = base_channels
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.mask_dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.mask_dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.mask_dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.mask_dec1 = DecoderBlock(c * 2, c, c)

        self.skel_dec4 = DecoderBlock(c * 16, c * 8, c * 8)
        self.skel_dec3 = DecoderBlock(c * 8, c * 4, c * 4)
        self.skel_dec2 = DecoderBlock(c * 4, c * 2, c * 2)
        self.skel_dec1 = DecoderBlock(c * 2, c, c)

        self.mask_head = nn.Conv2d(c, 1, 1)
        self.skel_head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        m = self.mask_dec4(b, e4)
        m = self.mask_dec3(m, e3)
        m = self.mask_dec2(m, e2)
        m = self.mask_dec1(m, e1)

        s = self.skel_dec4(b, e4)
        s = self.skel_dec3(s, e3)
        s = self.skel_dec2(s, e2)
        s = self.skel_dec1(s, e1)

        return torch.sigmoid(self.mask_head(m)), torch.sigmoid(self.skel_head(s))
