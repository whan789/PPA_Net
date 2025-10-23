import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN
from layers.Patch_Summary import PatchSummary
from layers.Spectral_Block import SpectralBlock
from layers.Patch_Aggregator import PatchAggregator

    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.kd_lambda = configs.kd_lambda
        self.use_teacher = configs.use_teacher
        self.rank = configs.rank
        self.modes = configs.modes
        self.revin = RevIN(self.enc_in)
        self.hidden_dim = configs.kd_hidden_dim
        self.top_k = configs.top_k

        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = math.floor((self.seq_len - self.patch_len) / self.stride) + 1
        self.patch_embedding = nn.Linear(self.patch_len, self.d_model)
        
        self.patch_agg = PatchAggregator(self.enc_in, self.d_model, Nk=self.patch_num)

        self.patch_summary = PatchSummary(configs, self.seq_len, self.pred_len,
                                             self.patch_len, self.stride, self.d_model, self.rank)
        if self.use_teacher:
            self.spectral_block = SpectralBlock(self.enc_in, self.modes, self.top_k)
            self.proj_t = nn.Linear(self.seq_len, self.hidden_dim)
            self.proj_s = nn.Linear(self.d_model, self.hidden_dim)

    def patching(self, x):
        return x.unfold(2, self.patch_len, self.stride)

    def forward(self, x_enc, x_mark=None, x_dec=None, x_mark_dec=None,
                mask=None, return_loss=False, **kwargs):
        
        x = self.revin(x_enc, mode='norm')
        x = x.permute(0, 2, 1)

        patches = self.patching(x)

        # Project the patch_len dimension into the model dimension
        patch_latent = self.patch_embedding(patches)

        # Patch which summarizes the context of min, max, norm representations
        agg_patch = self.patch_agg(patch_latent, patches) # (B, C, 1, D)

        # Final output from Patch Branch
        y_patch = self.patch_summary(patch_latent, agg_patch)  # (B, C, pred_len)

        if self.use_teacher and return_loss:
            y_freq = self.spectral_block(x)
            teacher = self.proj_t(y_freq) 
            # Student : the output of PPA module            
            student = self.proj_s(agg_patch.squeeze(2))
            # knowledge distillation loss for aligning the representation of PPA module with frequency block
            kd_loss = self.kd_lambda * F.mse_loss(student, teacher.detach())
        
        else:
            kd_loss = torch.tensor(0.0, device=y_patch.device)

        output = y_patch.permute(0, 2, 1)
        output = self.revin(output, mode='denorm')

        return output[:, -self.pred_len:, :], kd_loss