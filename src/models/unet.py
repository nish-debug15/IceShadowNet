"""
src/models/unet.py
==================
Lightweight 4-level U-Net with Dropout2d in bottleneck and decoder stages
for pixel-level lunar SAR polarimetric anomaly segmentation (and MC Dropout
uncertainty estimation in Block 3).

Input: (B, 3, 64, 64) raw radar log-intensities [HH, HV, VV].
Output: (B, 1, 64, 64) raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IceUNet(nn.Module):
    """
    Lightweight 4-level U-Net (base_ch=24 -> 24, 48, 96, 192).
    Dropout2d is placed in the bottleneck and decoder blocks so MC Dropout
    can be enabled at inference by keeping Dropout2d layers in train() mode.
    """

    def __init__(self, in_channels: int = 3, base_ch: int = 24, dropout_p: float = 0.15):
        super().__init__()
        self.in_channels = in_channels
        self.base_ch = base_ch
        self.dropout_p = dropout_p

        # Encoder (64 -> 32 -> 16 -> 8)
        self.enc1 = ConvBlock(in_channels, base_ch, dropout_p=0.0)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, dropout_p=0.0)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, dropout_p=0.0)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck (8x8) with Dropout2d
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8, dropout_p=dropout_p)

        # Decoder with Dropout2d
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4, dropout_p=dropout_p)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2, dropout_p=dropout_p)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch, dropout_p=0.0)

        self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.head(d1)
