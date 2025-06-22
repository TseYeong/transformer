import math
from typing import Optional, Tuple

import torch
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    Compute scaled dot product attention
    """
    def __init__(self, dropout: float = 0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self,
            q: torch.Tensor,  # [B, H, Lq, D]
            k: torch.Tensor,  # [B, H, Lk, D]
            v: torch.Tensor,  # [B, H, Lv, D]
            mask: Optional[torch.Tensor] = None,  # [B, 1, Lq, Lk]
            position_bias: Optional[torch.Tensor] = None,  # [1, H, Lq, Lkv]
            eps: Optional[float] = 1e-12  # Prevent zero division
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k + eps)

        if position_bias is not None:
            scores = scores + position_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn
