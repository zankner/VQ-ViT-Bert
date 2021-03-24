import os
import urllib
from pathlib import Path
from tqdm import tqdm
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# constants
CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'


# helpers methods
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))


def map_pixels(x, eps=0.1):
    return (1 - 2 * eps) * x + eps


def unmap_pixels(x, eps=0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)


def download(url, filename=None, root=CACHE_PATH):
    os.makedirs(root, exist_ok=True)
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target_tmp,
                                                     "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")),
                  ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    return download_target


# pretrained Discrete VAE from OpenAI
class OpenAIDiscreteVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
        self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))

        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc(img)
        z = torch.argmax(z_logits, dim=1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h=int(sqrt(n)))

        z = F.one_hot(img_seq, num_classes=self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        raise NotImplemented
