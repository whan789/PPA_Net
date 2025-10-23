import math
import torch
import torch.nn as nn
from layers.PatchMLPBlock import PatchMLPBlock
from layers.FlattenHead import FlattenHead

class PatchSummary(nn.Module):
    '''
    Synthesizes information from patch_latent (patch representations)
    and agg_patch (the output of PPA module) to derive the final output of the Patch Branch)
    '''
    def __init__(self, configs, seq_len, pred_len, patch_len, stride, d_model, rank):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = math.floor(self.seq_len - self.patch_len) // self.stride + 1
        self.d_model = d_model

        self.mlp_blocks = nn.ModuleList([
            PatchMLPBlock(num_patches=self.patch_num+1,
                          d_model=self.d_model,
                          hidden_dim=self.d_model,
                          dropout=configs.dropout,
                          rank=rank)
            for _ in range(configs.e_layers)
        ])

        self.head_nf = self.d_model * (self.patch_num + 1)
        self.head = FlattenHead(self.head_nf, self.pred_len, head_dropout=configs.dropout)

    def forward(self, patch_latent, agg_patch):
        B, C, N, D = patch_latent.shape

        total_patch = torch.cat([patch_latent, agg_patch], dim=2)  # [B, C, N+1, D]

        x = total_patch.reshape(B * C, N+1, D)
        for block in self.mlp_blocks:
            x = block(x)
        x = x.reshape(B, C, N+1, D).permute(0, 1, 3, 2)  # [B, C, D, N+1]

        dec_out = self.head(x)  # [B, C, pred_len]
        return dec_out