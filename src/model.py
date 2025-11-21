#TODO: Implement your model
import torch
import torch.nn as nn


# -------------------------------------
# Basic Double Conv Block
# -------------------------------------
class DoubleConv(nn.Module):
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


# -------------------------------------
# Up Block = ConvTranspose + DoubleConv
# -------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)  # concat doubles channels

    def forward(self, x, skip):
        x = self.up(x)

        # match size if odd shapes
        if x.size()[2:] != skip.size()[2:]:
            x = nn.functional.interpolate(x, size=skip.size()[2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# -------------------------------------
# Heavy UNet (deep 5-level)
# -------------------------------------
class UNetHeavy(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        # Encoder (down path)
        self.e1 = DoubleConv(in_ch, 64)
        self.p1 = nn.MaxPool2d(2)

        self.e2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.e3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.e4 = DoubleConv(256, 512)
        self.p4 = nn.MaxPool2d(2)

        # Bottleneck
        self.b = DoubleConv(512, 1024)

        # Decoder (up path)
        self.u4 = UpBlock(1024, 512)
        self.u3 = UpBlock(512, 256)
        self.u2 = UpBlock(256, 128)
        self.u1 = UpBlock(128, 64)

        # Final output
        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        e4 = self.e4(self.p3(e3))

        # Bottleneck
        b = self.b(self.p4(e4))

        # Decoder
        d4 = self.u4(b, e4)
        d3 = self.u3(d4, e3)
        d2 = self.u2(d3, e2)
        d1 = self.u1(d2, e1)

        return self.out(d1)


def UNetSmall(in_ch=1, out_ch=1, base=16):
    """Compatibility function to not break train.py"""
    return UNetHeavy(in_ch, out_ch)
