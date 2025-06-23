import torch
import torch.nn as nn

from .attention import BaseAttention


def _build_rotary(x):
    x1 = x.unsqueeze(0).unsqueeze(0)
    return torch.stack([x1, x1], dim=-1).flatten(-2)


class RoPEPositionEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** torch.arange(0, dim, 2).float() / dim)
        positions = torch.arange(0, max_len).float()
        sinusoid = torch.einsum("i,j->ij", positions, inv_freq)  # [max_len, dim / 2]

        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)

        self.register_buffer("sin", _build_rotary(sin), persistent=False)
        self.register_buffer("cos", _build_rotary(cos), persistent=False)

    def forward(self, x, seq_len=None):
        seq_len = seq_len or x.size(2)
        sin = self.sin[:, :, :seq_len, :]
        cos = self.cos[:, :, :seq_len, :]
        x1, x2 = x[..., ::2], x[..., 1::2]

        x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + x_rot * sin


class RoPEAttentino(BaseAttention):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__(d_model, n_heads, dropout)
        self.rope = RoPEPositionEncoding(self.d_head, max_len)

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:
            x_kv = x_q

        Q, K, V = self._reshape_qkv(x_q, x_kv)

        Q = self.rope(Q)
        K = self.rope(K)

        context, _ = self.attn(Q, K, V, mask)
        out = context.transpose(1, 2).contiguous().view(x_q.size(0), -1, self.d_model)

        return out
