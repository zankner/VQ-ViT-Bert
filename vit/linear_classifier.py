import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self,
                 transformer,
                 embedding_dim,
                 out_dim,
                 freeze_transformer=True):
        super(LinearClassifier, self).__init__()

        # Set transformer and freeze parameters
        self.transformer = transformer
        self.freeze_transformer = freeze_transformer
        if self.freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Create linear classification head
        self.classification_head = nn.Linear(embedding_dim, out_dim)

    def forward(self, x):
        embedding = self.transformer(x, compute_codebook=True)
        cls_embedding = embedding[:, 0]
        out = self.classification_head(cls_embedding)

        return out
