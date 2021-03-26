import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim,
                                           dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    Residual(
                        PreNorm(
                            dim,
                            Attention(dim,
                                      heads=heads,
                                      dim_head=dim_head,
                                      dropout=dropout))),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim,
                                                 dropout=dropout)))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self,
                 vae,
                 dim,
                 depth,
                 heads,
                 mlp_dim,
                 vocab_size,
                 num_codebook_indeces,
                 dim_head=64,
                 dropout=0.0,
                 emb_dropout=0.0,
                 cls_token_id=1,
                 pad_token_id=0):
        super(ViT, self).__init__()

        # VAE to get codebook indeces
        self.vae = vae
        self.vae.requires_grad = False

        # Token and positional embedding
        self.embedding = nn.Embedding(vocab_size,
                                      dim,
                                      padding_idx=pad_token_id)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer body of ViT
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)

        # Special tokens
        self.cls_token_id = cls_token_id
        self.num_special_tokens = vocab_size - num_codebook_indeces

    def forward(self, x, compute_codebook=False):
        device = x.device

        if compute_codebook:
            codebook_indeces = self.vae.get_codebook_indices(x)
            codebook_indeces += self.num_special_tokens
            cls_tokens = torch.full((len(x), 1),
                                    self.cls_token_id,
                                    device=device)
            x = torch.cat([cls_tokens, codebook_indeces], dim=1).long()

        b, n = x.shape

        x = self.embedding(x)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        return x