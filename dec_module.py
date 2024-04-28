import torch
import torch.nn as nn
import torch.nn.functional as F
from att_module import PointAttentionBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up3 = UpConvBlock(512, 256)
        self.up2 = UpConvBlock(256, 128)
        self.up1 = UpConvBlock(128, 64)

        self.att3 = PointAttentionBlock(F_g=256, F_l=256, F_int=128)
        self.att2 = PointAttentionBlock(F_g=128, F_l=128, F_int=64)
        self.att1 = PointAttentionBlock(F_g=64, F_l=64, F_int=32)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)
        p1 = self.pool(x1)
        x2 = self.enc2(p1)
        p2 = self.pool(x2)
        x3 = self.enc3(p2)
        p3 = self.pool(x3)
        x4 = self.enc4(p3)
        
        # Decoder path with attention gates
        g3 = self.up3(x4, self.att3(x4, x3))
        g2 = self.up2(g3, self.att2(g3, x2))
        g1 = self.up1(g2, self.att1(g2, x1))
        
        # Final output layer
        output = self.final_conv(g1)
        return output

