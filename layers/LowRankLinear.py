import torch.nn as nn

class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, bias=True):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=bias)

    def forward(self, x):
        return self.B(self.A(x))