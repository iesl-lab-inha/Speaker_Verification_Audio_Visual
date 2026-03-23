import torch
import torch.nn as nn
from utils import lengths_to_mask


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=5):
        super().__init__()
        self.num_heads = num_heads
        self.attn = nn.Linear(input_dim, num_heads)

    def forward(self, x, lengths=None):
        """
        x: [B, T, D]
        returns: [B, H*D]
        """
        logits = self.attn(x)

        if lengths is not None:
            mask = lengths_to_mask(lengths, x.size(1))
            mask = ~mask
            logits = logits.masked_fill(mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(logits, dim=1)

        pooled = []
        for h in range(self.num_heads):
            w = weights[:, :, h].unsqueeze(-1)
            pooled_h = torch.sum(x * w, dim=1)
            pooled.append(pooled_h)

        return torch.cat(pooled, dim=-1)
