import torch
from vit import ViT, MPP

v = ViT(dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        vocab_size=256,
        embedding_dim=512,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0)

m = MPP(transformer=v,
        mask_prob=0.15,
        replace_prob=0.5,
        num_tokens=256,
        random_token_prob=0.3,
        mask_token_id=2,
        pad_token_id=0,
        mask_ignore_token_ids=[])

x = torch.randint(0, 255, (1, 10))
out = m(x)

print(out)