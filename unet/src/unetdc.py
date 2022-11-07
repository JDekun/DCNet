from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x, x1


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )
        

class Projector(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Projector, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU()
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )


class DC_UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64,
                 proj_d: int = 128):
        super(DC_UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        self.proj_conv1 = Projector(base_c, proj_d)
        self.proj_conv2 = Projector(base_c * 2, proj_d)
        self.proj_conv3 = Projector(base_c * 4, proj_d)
        self.proj_conv4 = Projector(base_c * 8, proj_d)

        self.out_conv1 = OutConv(proj_d, num_classes)
        self.out_conv2 = OutConv(proj_d, num_classes)
        self.out_conv3 = OutConv(proj_d, num_classes)
        self.out_conv4 = OutConv(proj_d, num_classes)

    def forward(self, x: torch.Tensor, epoch: int = 0, with_contrast : int = 40 ) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        # ↓↓↓↓↓ #
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # ↑↑↑↑↑
        x5 = self.down4(x4)
        # ↓↓↓↓↓ #
        x, y4 = self.up1(x5, x4)
        x, y3 = self.up2(x, x3)
        x, y2 = self.up3(x, x2)
        # ↑↑↑↑↑ #
        x, y1 = self.up4(x, x1)


        logits = self.out_conv(x)

        if with_contrast < 0:
            return {"out": logits}
        elif with_contrast > epoch:
            return {"out": logits}
        else:
            px1 = F.normalize(self.proj_conv1(x1), p=2, dim=1)
            px2 = F.normalize(self.proj_conv2(x2), p=2, dim=1)
            px3 = F.normalize(self.proj_conv3(x3), p=2, dim=1)
            px4 = F.normalize(self.proj_conv4(x4), p=2, dim=1)

            py1 = F.normalize(self.proj_conv1(y1), p=2, dim=1)
            py2 = F.normalize(self.proj_conv2(y2), p=2, dim=1)
            py3 = F.normalize(self.proj_conv3(y3), p=2, dim=1)
            py4 = F.normalize(self.proj_conv4(y4), p=2, dim=1)

            Label1 = self.out_conv1(py1)
            Label2 = self.out_conv2(py2)
            Label3 = self.out_conv3(py3)
            Label4 = self.out_conv4(py4)
            
            return {"out": logits, 
                    "L1": [px1.detach(), py1, Label1], 
                    "L2": [px2.detach(), py2, Label2], 
                    "L3": [px3.detach(), py3, Label3], 
                    "L4": [px4.detach(), py4, Label4]}