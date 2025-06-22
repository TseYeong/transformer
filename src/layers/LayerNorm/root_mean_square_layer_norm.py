import torch
from torch import nn
from .basicLM import BaseLayerNorm


class RMSNorm(BaseLayerNorm):
    """
    Root Mean Square Layer Normalization.
    Applies normalization across the last dimension and scales the output.
    """

    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        rms = x.norm(p=2, dim=-1, keepdim=True) / (self.d_model ** 0.5)
        x_norm = x / (rms + self.eps)
        return self.gamma * x_norm
