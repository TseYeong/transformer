import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from src.layers.scale_dot_product_attention import ScaleDotProductAttention


class BaseAttention(nn.Module, ABC):
    def __init__(self, d_model: int, n_heads: int, dropout: int = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.attn = ScaleDotProductAttention(dropout)

    @abstractmethod
    def forward(self, x_q, x_kv):
        """
        :param x_q: [B, Lq, D]
        :type x_q: torch.Tensor
        :param x_kv: [B, Lkv, D] or None (self-attn)
        :type x_kv: Union[torch.Tensor | None]
        :return: [B, Lq, D]
        :rtype: torch.Tensor
        """
        pass

    def _reshape_qkv(self, x_q: torch.Tensor, x_kv: torch.Tensor):
        B, Lq, _ = x_q.size()
        Lkv = x_kv.size(1)

        Q = self.q_proj(x_q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x_kv).view(B, Lkv, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x_kv).view(B, Lkv, self.n_heads, self.d_head).transpose(1, 2)

        return Q, K, V
