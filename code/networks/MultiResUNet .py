# Copyright 2025
# Adapted for MultiResUNet architecture based on:
# Ibtehaz, N., & Rahman, M. S. (2020). MultiResUNet: Rethinking the U-Net architecture for multimodal biomedical image segmentation.
#
# This file follows a similar structural style to RecursiveUNet.py
# but implements the MultiResUNet architecture for 2D segmentation.

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Basic Conv-BN-ReLU block
# -------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# -------------------------------------------------
# MultiRes Block
# Approximates multi-scale convs with stacked 3x3s
# -------------------------------------------------
class MultiResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.67):
        """
        in_channels: number of input channels
        out_channels: nominal output channels at this level
        alpha: width multiplier from the paper
        """
        super(MultiResBlock, self).__init__()

        W = int(alpha * out_channels)

        c1 = int(W * 0.167)
        c2 = int(W * 0.333)
        c3 = int(W * 0.5)

        self.conv3x3 = ConvBNReLU(in_channels, c1, kernel_size=3, padding=1)
        self.conv5x5 = ConvBNReLU(c1, c2, kernel_size=3, padding=1)
        self.conv7x7 = ConvBNReLU(c2, c3, kernel_size=3, padding=1)

        self.batch_norm = nn.BatchNorm2d(c1 + c2 + c3)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 shortcut to match channels
        self.shortcut_conv = nn.Conv2d(
            in_channels, c1 + c2 + c3, kernel_size=1, bias=False
        )
        self.shortcut_bn = nn.BatchNorm2d(c1 + c2 + c3)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x1)
        x3 = self.conv7x7(x2)

        out = torch.cat([x1, x2, x3], dim=1)
        out = self.batch_norm(out)

        shortcut = self.shortcut_bn(self.shortcut_conv(x))

        out = self.relu(out + shortcut)
        return out


# -------------------------------------------------
# ResPath
# Residual path used along skip connections
# -------------------------------------------------
class ResPath(nn.Module):
    def __init__(self, in_channels, out_channels, length):
        """
        length: how many residual blocks in this path
        """
        super(ResPath, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.shortcut_convs = nn.ModuleList()
        self.shortcut_bns = nn.ModuleList()

        for i in range(length):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = out_channels

            self.convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=False)
            )
            self.bns.append(nn.BatchNorm2d(out_channels))

            self.shortcut_convs.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False)
            )
            self.shortcut_bns.append(nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        for conv, bn, sc_conv, sc_bn in zip(
            self.convs, self.bns, self.shortcut_convs, self.shortcut_bns
        ):
            shortcut = sc_bn(sc_conv(out))
            out = bn(conv(out))
            out = self.relu(out + shortcut)
        return out


# -------------------------------------------------
# MultiResUNet main architecture
# -------------------------------------------------
class MultiResUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=3,
        base_filters=32,
        alpha=1.67,
    ):
        """
        in_channels: 1 for ultrasound
        num_classes: segmentation classes
        base_filters: base number of feature maps (32 or 64)
        alpha: width multiplier for MultiRes blocks
        """
        super(MultiResUNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.alpha = alpha

        # ---------------- Encoder ----------------
        # Helper to compute actual output channels from MultiResBlock
        def get_output_channels(out_channels, alpha):
            W = int(alpha * out_channels)
            c1 = int(W * 0.167)
            c2 = int(W * 0.333)
            c3 = int(W * 0.5)
            return c1 + c2 + c3
        
        self.mresblock1 = MultiResBlock(in_channels, base_filters, alpha=alpha)
        self.pool1 = nn.MaxPool2d(2)
        out1 = get_output_channels(base_filters, alpha)

        self.mresblock2 = MultiResBlock(out1, base_filters * 2, alpha=alpha)
        self.pool2 = nn.MaxPool2d(2)
        out2 = get_output_channels(base_filters * 2, alpha)

        self.mresblock3 = MultiResBlock(out2, base_filters * 4, alpha=alpha)
        self.pool3 = nn.MaxPool2d(2)
        out3 = get_output_channels(base_filters * 4, alpha)

        self.mresblock4 = MultiResBlock(out3, base_filters * 8, alpha=alpha)
        self.pool4 = nn.MaxPool2d(2)
        out4 = get_output_channels(base_filters * 8, alpha)

        # ---------------- Bottleneck ----------------
        self.mresblock5 = MultiResBlock(out4, base_filters * 16, alpha=alpha)
        out5 = get_output_channels(base_filters * 16, alpha)

        # ---------------- ResPaths for skips ----------------
        self.respath1 = ResPath(
            in_channels=out1,
            out_channels=out1,
            length=4,
        )
        self.respath2 = ResPath(
            in_channels=out2,
            out_channels=out2,
            length=3,
        )
        self.respath3 = ResPath(
            in_channels=out3,
            out_channels=out3,
            length=2,
        )
        self.respath4 = ResPath(
            in_channels=out4,
            out_channels=out4,
            length=1,
        )

        # ---------------- Decoder ----------------
        self.up4 = nn.ConvTranspose2d(
            out5,
            out4,
            kernel_size=2,
            stride=2,
        )
        self.mresblock_up4 = MultiResBlock(
            out4 * 2, base_filters * 8, alpha=alpha
        )
        out_up4 = get_output_channels(base_filters * 8, alpha)

        self.up3 = nn.ConvTranspose2d(
            out_up4,
            out3,
            kernel_size=2,
            stride=2,
        )
        self.mresblock_up3 = MultiResBlock(
            out3 * 2, base_filters * 4, alpha=alpha
        )
        out_up3 = get_output_channels(base_filters * 4, alpha)

        self.up2 = nn.ConvTranspose2d(
            out_up3,
            out2,
            kernel_size=2,
            stride=2,
        )
        self.mresblock_up2 = MultiResBlock(
            out2 * 2, base_filters * 2, alpha=alpha
        )
        out_up2 = get_output_channels(base_filters * 2, alpha)

        self.up1 = nn.ConvTranspose2d(
            out_up2,
            out1,
            kernel_size=2,
            stride=2,
        )
        self.mresblock_up1 = MultiResBlock(
            out1 * 2, base_filters, alpha=alpha
        )
        out_up1 = get_output_channels(base_filters, alpha)

        # ---------------- Final classifier ----------------
        self.final_conv = nn.Conv2d(
            out_up1, num_classes, kernel_size=1
        )

    def forward(self, x):
        # -------- Encoder --------
        x1 = self.mresblock1(x)   # ~ alpha*F
        p1 = self.pool1(x1)

        x2 = self.mresblock2(p1)  # ~ alpha*2F
        p2 = self.pool2(x2)

        x3 = self.mresblock3(p2)  # ~ alpha*4F
        p3 = self.pool3(x3)

        x4 = self.mresblock4(p3)  # ~ alpha*8F
        p4 = self.pool4(x4)

        # -------- Bottleneck --------
        x5 = self.mresblock5(p4)  # ~ alpha*16F

        # -------- ResPaths (skip branches) --------
        r1 = self.respath1(x1)
        r2 = self.respath2(x2)
        r3 = self.respath3(x3)
        r4 = self.respath4(x4)

        # -------- Decoder --------
        u4 = self.up4(x5)
        u4 = torch.cat([u4, r4], dim=1)
        u4 = self.mresblock_up4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, r3], dim=1)
        u3 = self.mresblock_up3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, r2], dim=1)
        u2 = self.mresblock_up2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, r1], dim=1)
        u1 = self.mresblock_up1(u1)

        out = self.final_conv(u1)   # logits [B, num_classes, H, W]
        return out


__all__ = ["MultiResUNet"]
