# ===== unet_model.py =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_part_temp import DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)

