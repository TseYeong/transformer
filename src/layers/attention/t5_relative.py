import math

import torch
import torch.nn as nn


class T5RelativeBias(nn.Module):
    """
    Implements T5-style relative position bias.
    """
    def __init__(self, n_heads: int, n_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        assert n_buckets % 2 == 0, "Buckets number should be even"
        self.relative_attention_bias = nn.Embedding(n_buckets, n_heads)
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance

    def _relative_position_bucket(self, relative_positions: torch.Tensor) -> torch.Tensor:
        """
        Assigns relative position into buckets as per T5 logic.
        """
        n_buckets = self.n_buckets
        max_distance = self.max_distance

        n = relative_positions
        ret = 0

        is_small = (n.abs() < n_buckets // 2)
        val_if_small = n.abs()

        log_scale = (
            (n.abs().float() / max_distance + 1e-6).log()
            / math.log(128.0 / (n_buckets // 2))
        ).long()
        val_if_large = n_buckets // 2 + log_scale
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, n_buckets - 1))

        final = torch.where(is_small, val_if_small, val_if_large)
        final = torch.where(n < 0, final, final + n_buckets // 2)
        return final

    def forward(
        self,
        q_len: int,
        k_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        context_position = torch.arange(q_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(k_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # [q_len, k_len]

        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)  # [q_len, k_len, num_heads]
        values = values.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, q_len, k_len]

        return values
