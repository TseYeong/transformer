from .attention import BaseAttention
from .t5_relative import T5RelativeBias


class T5RelativeAttention(BaseAttention):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1,
            n_buckets: int = 32,
            max_distance: int = 128
    ):
        super().__init__(d_model, n_heads, dropout)
        self.bias_layer = T5RelativeBias(n_heads, n_buckets, max_distance)

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:
            x_kv = x_q

        B, Lq, _ = x_q.shape
        Lkv = x_kv.shape[1]

        Q, K, V = self._reshape_qkv(x_q, x_kv)  # [B, H, L, D]
        pos_bias = self.bias_layer(Lq, Lkv, x_q.device)

        context, _ = self.attn(Q, K, V, mask, pos_bias)
        out = context.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # [B, H, Lq, D]

        return self.o_proj(out)
