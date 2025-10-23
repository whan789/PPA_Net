import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralBlock(nn.Module):
    def __init__(self, channels, modes=16, top_k=16, kernel_size=3, n_heads=1, dropout=0.0,dilated_kernel=4, dilation=2):  #3,2
        super().__init__()
        self.channels = channels
        self.modes = modes
        self.top_k = top_k 

        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels
        )

        self.attn = nn.MultiheadAttention(embed_dim=top_k, num_heads=n_heads,
                                           dropout=dropout, batch_first=True)
        self.hidden_dim = self.top_k * 4
        
        # Dilated CNN 
        self.dilated_conv = nn.Conv1d(
            channels, channels,
            kernel_size=dilated_kernel,
            dilation=dilation,
            padding=dilation * (dilated_kernel - 1) // 2,  # SAME padding
            groups=channels
        )

    def forward(self, x): 
        B, C, T = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)    # [B, C, T//2+1]
        out_ft = torch.zeros_like(x_ft)

        # Select only the low-frequency components (modes)
        low_freq = x_ft[:, :, :self.modes]  # [B, C, modes]

        real, imag = low_freq.real, low_freq.imag

        # Select the top-k dominant frequencies from the low-frequency components
        # The rest parts will be passed through (identity mapping)
        real_topk, real_rest = real[:, :, :self.top_k], real[:, :, self.top_k:]
        imag_topk, imag_rest = imag[:, :, :self.top_k], imag[:, :, self.top_k:]
             
        real_topk = real_topk.view(B * C, 1, self.top_k)   
        imag_topk = imag_topk.view(B * C, 1, self.top_k)
        
        # Apply self-attention to weight the importance of the top-k frequencies
        real_attn, _ = self.attn(real_topk, real_topk, real_topk)  
        imag_attn, _ = self.attn(imag_topk, imag_topk, imag_topk)

        # reshape [B*C, top_k] -> [B, C, top_k]
        real_attn = real_attn.view(B, C, self.top_k)
        imag_attn = imag_attn.view(B, C, self.top_k)
        
        real_fused = torch.cat([real_attn, real_rest], dim=-1)
        imag_fused = torch.cat([imag_attn, imag_rest], dim=-1)

        out_low = torch.complex(real_fused, imag_fused)
        out_ft[:, :, :self.modes] = out_low

        # Apply IFFT to convert back to the time domain
        x_filtered = torch.fft.irfft(out_ft, n=T, dim=-1)
        x_filtered = self.dilated_conv(x_filtered) 
        return x_filtered