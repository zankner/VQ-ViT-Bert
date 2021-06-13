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
    def __init__(self,
                 embedding_dim,
                 num_blocks=2,
                 feature_dim=64,
                 channels=3):
        super(Encoder, self).__init__()

        # Input conv
        self.input_conv = nn.Conv2d(channels,
                                    feature_dim,
                                    kernel_size=7,
                                    padding=3)

        # Blocks
        self.blocks = nn.Sequential(
            OrderedDict([
                ('group_1',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(1 * feature_dim, 1 * feature_dim))
                           for i in range(num_blocks)],
                         ('pool', nn.MaxPool2d(kernel_size=2))
                     ]))),
                ('group_2',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(
                                1 * feature_dim if i == 0 else 2 * feature_dim,
                                2 * feature_dim)) for i in range(num_blocks)],
                         ('pool', nn.MaxPool2d(kernel_size=2))
                     ]))),
                # ('group_3',
                #  nn.Sequential(
                #      OrderedDict([
                #          *[(f'block_{i+1}',
                #             ResBlock(
                #                 2 * feature_dim if i == 0 else 4 * feature_dim,
                #                 4 * feature_dim)) for i in range(num_blocks)],
                #          ('pool', nn.MaxPool2d(kernel_size=2))
                #      ]))),
                # ('group_4',
                #  nn.Sequential(
                #      OrderedDict([
                #          *[(f'block_{i+1}',
                #             ResBlock(
                #                 4 * feature_dim if i == 0 else 8 * feature_dim,
                #                 8 * feature_dim)) for i in range(num_blocks)],
                #          ('pool', nn.MaxPool2d(kernel_size=2))
                #      ])))
            ]))

        # Output conv
        self.output_conv = nn.Conv2d(2 * feature_dim,
                                     embedding_dim,
                                     kernel_size=1)

        # Activations
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.relu(x)

        x = self.blocks(x)

        x = self.output_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_blocks=2,
                 feature_dim=64,
                 channels=3):
        super(Decoder, self).__init__()

        # Input conv
        self.input_conv = nn.Conv2d(embedding_dim,
                                    feature_dim * 2,
                                    kernel_size=1)

        # Blocks
        self.blocks = nn.Sequential(
            OrderedDict([
                # ('group_1',
                #  nn.Sequential(
                #      OrderedDict([
                #          *[(f'block_{i+1}',
                #             ResBlock(8 * feature_dim, 8 * feature_dim))
                #            for i in range(num_blocks)],
                #          ('upsample',
                #           nn.Upsample(scale_factor=2, mode="nearest"))
                #      ]))),
                # ('group_2',
                #  nn.Sequential(
                #      OrderedDict([
                #          *[(f'block_{i+1}',
                #             ResBlock(
                #                 8 * feature_dim if i == 0 else 4 * feature_dim,
                #                 4 * feature_dim)) for i in range(num_blocks)],
                #          ('upsample',
                #           nn.Upsample(scale_factor=2, mode="nearest"))
                #      ]))),
                ('group_3',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(
                                2 * feature_dim if i == 0 else 2 * feature_dim,
                                2 * feature_dim)) for i in range(num_blocks)],
                         ('upsample',
                          nn.Upsample(scale_factor=2, mode="nearest"))
                     ]))),
                ('group_4',
                 nn.Sequential(
                     OrderedDict([
                         *[(f'block_{i+1}',
                            ResBlock(
                                2 * feature_dim if i == 0 else 1 * feature_dim,
                                1 * feature_dim)) for i in range(num_blocks)],
                         ('upsample',
                          nn.Upsample(scale_factor=2, mode="nearest"))
                     ])))
            ]))

        # Output conv
        self.output_conv = nn.Conv2d(1 * feature_dim, channels, kernel_size=1)

        # Activations
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.relu(x)

        x = self.blocks(x)

        x = self.output_conv(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_codebook_indeces,
                 embedding_dim,
                 commitment_cost=0.6):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_codebook_indeces = num_codebook_indeces

        self._embedding = nn.Embedding(self._num_codebook_indeces,
                                       self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_codebook_indeces,
                                             1 / self._num_codebook_indeces)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, 'b c h w -> b h w c')
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.contiguous().view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) +
                     torch.sum(self._embedding.weight**2, dim=1) -
                     2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],
                                self._num_codebook_indeces,
                                device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings,
                                 self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                                          torch.log(avg_probs + 1e-10)))
        # convert encoding indices from (BHW)C -> B(HWC)
        encoding_indices = rearrange(encoding_indices,
                                     "(b h w) c -> b (h w c)",
                                     h=input_shape[1],
                                     w=input_shape[2])

        # convert quantized from BHWC -> BCHW
        quantized = rearrange(quantized, "b h w c -> b c h w")
        return loss, quantized.contiguous(), perplexity, encoding_indices


class VQVae(nn.Module):
    def __init__(self,
                 num_codebook_indeces,
                 embedding_dim,
                 num_blocks=2,
                 feature_dim=64,
                 channels=3,
                 commitment_cost=0.6):
        super(VQVae, self).__init__()

        # Encoder and decoder
        self.encoder = Encoder(embedding_dim, num_blocks, feature_dim,
                               channels)
        self.decoder = Decoder(embedding_dim, num_blocks, feature_dim,
                               channels)

        # Vector quantizer
        self.vector_quantizer = VectorQuantizer(num_codebook_indeces,
                                                embedding_dim, commitment_cost)

    def forward(self, x):
        x = self.encoder(x)
        vq_loss, vq_z, perplexity, _ = self.vector_quantizer(x)
        out = self.decoder(vq_z)

        return out, vq_loss, perplexity

    @torch.no_grad()
    def get_codebook_indices(self, x):
        x = self.encoder(x)
        _, _, _, encodings = self.vector_quantizer(x)
        return encodings