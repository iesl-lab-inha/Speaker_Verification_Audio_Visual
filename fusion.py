import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, audio_x, visual_x):
        """
        audio_x: [B, T, D]
        visual_x: [B, T, D]
        """
        g = self.gate(torch.cat([audio_x, visual_x], dim=-1))
        return g * audio_x + (1.0 - g) * visual_x
