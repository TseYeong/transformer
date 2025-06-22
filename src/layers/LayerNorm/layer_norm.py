import torch
from torch import nn
from .basicLM import BaseLayerNorm


class LayerNorm(BaseLayerNorm):
    """
    Implementation of layer norm
    """

    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbias=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * out + self.beta
