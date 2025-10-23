import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchAggregator(nn.Module):
    def __init__(self, C, D, Nk):
        super().__init__()
        self.query = nn.Parameter(torch.randn(C, 1, D))
        self.attn_normal = nn.MultiheadAttention(embed_dim=D, num_heads=1, batch_first=True)
        self.norm_normal = nn.LayerNorm(D)

        # min/max projection
        self.max_proj = nn.Linear(1, D)
        self.min_proj = nn.Linear(1, D)

        # Channel-wise weights (normal, max, min)
        self.alpha = nn.Parameter(torch.randn(C, 3))

        self.C = C
        self.D = D
        self.Nk = Nk

    def forward(self, patch_latent, patches):
        B, C, Nk, D = patch_latent.shape

        patch_latent_reshaped = patch_latent.reshape(B * C, Nk, D)
        q = self.query.repeat(B, 1, 1) 

        # Attention-based summary (normal features)
        pooled_norm, _ = self.attn_normal(q, patch_latent_reshaped, patch_latent_reshaped)
        pooled_norm = self.norm_normal(pooled_norm) 
        pooled_norm = pooled_norm.view(B, C, 1, D)

        # Extract the min, max value per patch
        max_val = patches.max(dim=-1, keepdim=True)[0] 
        min_val = patches.min(dim=-1, keepdim=True)[0]  

        # Projection per patch
        max_emb = self.max_proj(max_val)  
        min_emb = self.min_proj(min_val) 

        pooled_norm = pooled_norm.expand(-1, -1, Nk, -1) 

        pooled_stack = torch.stack([pooled_norm, max_emb, min_emb], dim=-1)  

        score = F.softmax(self.alpha, dim=-1)  
        weighted = (pooled_stack * score.view(1, C, 1, 1, 3)).sum(dim=-1)

        final = weighted.mean(dim=2, keepdim=True)

        return final