import torch.nn as nn
from layers.LowRankLinear import LowRankLinear

# PatchMLP Block (Low-rank MLP 적용)
class PatchMLPBlock(nn.Module):
    def __init__(self, num_patches, d_model, hidden_dim=32, dropout=0.0, rank=32):
        super().__init__()
        self.token_norm = nn.LayerNorm(num_patches)
        self.token_mixing = nn.Sequential(
            LowRankLinear(num_patches, hidden_dim, rank),
            nn.ReLU(),
            LowRankLinear(hidden_dim, num_patches, rank),
            nn.Dropout(dropout)
        )
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mixing = nn.Sequential(
            LowRankLinear(d_model, hidden_dim, rank),
            nn.ReLU(),
            LowRankLinear(hidden_dim, d_model, rank),
            nn.Dropout(dropout)
        )

    def forward(self, x):  # [B, N, D]
        x = x + self.token_mixing(self.token_norm(x.transpose(1, 2))).transpose(1, 2)
        x = x + self.channel_mixing(self.channel_norm(x))
        return x