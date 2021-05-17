import math
from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F

# helpers


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


# main class


class MPP(nn.Module):
    def __init__(self,
                 transformer,
                 vocab_size,
                 transformer_dim,
                 mask_prob=0.15,
                 replace_prob=0.9,
                 random_token_prob=0.,
                 cls_token_id=1,
                 mask_token_id=2,
                 pad_token_id=0,
                 mask_ignore_token_ids=[]):
        super().__init__()

        self.transformer = transformer

        # define linear head
        self.linear = nn.Linear(transformer_dim, vocab_size)

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_token_prob = random_token_prob

        self.vocab_size = vocab_size

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.mask_ignore_token_ids = set(
            [*mask_ignore_token_ids, pad_token_id, cls_token_id])
        self.num_special_tokens = self.transformer.num_special_tokens

    def forward(self, input, **kwargs):
        device = input.device

        # convert raw image to tokens
        codebook_indeces = self.transformer.vae.module.get_codebook_indices(
            input)
        codebook_indeces += self.num_special_tokens
        cls_tokens = torch.full((len(input), 1),
                                self.cls_token_id,
                                device=device)
        input = torch.cat([cls_tokens, codebook_indeces], dim=1).long()

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            assert self.vocab_size is not None, 'vocab_size keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = prob_mask_like(input, self.random_token_prob)
            random_tokens = torch.randint(0,
                                          self.vocab_size,
                                          input.shape,
                                          device=input.device)
            random_no_mask = mask_with_tokens(random_tokens,
                                              self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input[random_indices] = random_tokens[random_indices]

        # [mask] input
        replace_prob = prob_mask_like(input, self.replace_prob)
        masked_input = masked_input.masked_fill(mask * replace_prob,
                                                self.mask_token_id)

        # mask out any tokens to padding tokens that were not originally going to be masked
        labels = input.masked_fill(~mask, self.pad_token_id)

        # get generator output and get mlm loss
        transformer_out = self.transformer(masked_input, **kwargs)
        logits = self.linear(transformer_out)

        mlm_loss = F.cross_entropy(logits.transpose(1, 2),
                                   labels,
                                   ignore_index=self.pad_token_id)
        return mlm_loss
