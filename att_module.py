import torch
import torch.nn as nn
import torch.nn.functional as F

class PointAttentionLayer(nn.Module):
    def __init__(self, channels):
        super(PointAttentionLayer, self).__init__()
        # A convolution layer to create a single-channel output for attention weights
        self.conv = nn.Conv1d(channels, 1, kernel_size=1)  # Use 1x1 conv to learn spatial attention

    def forward(self, x):
        # Apply convolution to generate attention map
        attention_map = self.conv(x)
        # Use sigmoid to normalize the attention map to the range [0, 1]
        attention_weights = torch.sigmoid(attention_map)
        # Multiply the input by the attention weights
        return x * attention_weights


class PointAttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(PointAttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
