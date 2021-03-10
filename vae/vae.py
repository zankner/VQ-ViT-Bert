import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from collections import OrderedDict


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


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_blocks=2, feature_dim=64, channels=3):
        super(Encoder, self).__init__()

        # Input conv
        self.input_conv = nn.Conv2d(channels, feature_dim, kernel_size=7)

        # Blocks
        self.blocks = nn.Sequential(
            OrderedDict([
                ('group_1',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(1 * feature_dim, 1 * feature_dim))
                           for i in range(num_blocks)]
                     ]))),
                ('group_2',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(
                                1 * feature_dim if i == 0 else 2 * feature_dim,
                                2 * feature_dim)) for i in range(num_blocks)]
                     ]))),
                ('group_3',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(
                                2 * feature_dim if i == 0 else 4 * feature_dim,
                                4 * feature_dim)) for i in range(num_blocks)]
                     ]))),
                ('group_4',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(
                                4 * feature_dim if i == 0 else 8 * feature_dim,
                                8 * feature_dim)) for i in range(num_blocks)]
                     ])))
            ]))

        # Output conv
        self.output_conv = nn.Conv2d(8 * feature_dim,
                                     vocab_size,
                                     kernel_size=1)

        # Activations
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.relu(x)

        x = self.blocks(x)

        x = self.output_conv(x)

        return x
