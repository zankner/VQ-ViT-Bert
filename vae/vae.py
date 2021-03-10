import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock, self).__init__()

        # Skip connection
        self.skip = nn.Conv2d(
            in_features, out_features,
            kernel_size=1) if in_features != out_features else nn.Identity()

        # Conv layers
        self.conv1 = nn.Conv2d(in_features,
                               out_features,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(out_features,
                               out_features,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.skip(x)
        out = self.relu(out)

        return out
