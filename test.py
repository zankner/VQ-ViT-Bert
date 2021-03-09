import torch
from vit import ViT

v = ViT(dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_token=512,
        vocab_size=256,
        embedding_dim=512,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0)

x = torch.randint(0, 255, (2, 512))

out = v(x)

print(out.shape)