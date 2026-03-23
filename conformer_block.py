import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, dim, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.bn = nn.BatchNorm1d(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.ln(x).transpose(1, 2)
        y = self.pointwise1(y)
        y = torch.nn.functional.glu(y, dim=1)
        y = self.depthwise(y)
        y = self.bn(y)
        y = torch.nn.functional.silu(y)
        y = self.pointwise2(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class SharedConformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim, dropout=dropout)
        self.mha_ln = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.conv = ConvModule(dim, dropout=dropout)
        self.ffn2 = FeedForwardModule(dim, dropout=dropout)
        self.final_ln = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask=None):
        x = x + 0.5 * self.ffn1(x)

        y = self.mha_ln(x)
        y, _ = self.mha(y, y, y, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y

        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_ln(x)
        return x
