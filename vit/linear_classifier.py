import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, transformer, embedding_dim, out_dim):
        super(LinearClassifier, self).__init__()

        # Set transformer and linear head
        self.transformer = transformer
        self.classification_head = nn.Linear(embedding_dim, out_dim)

    def forward(self, x):
        embedding = self.transformer(x)
        cls_embedding = embedding[:, 0]
        out = self.classification_head(cls_embedding)

        return out
