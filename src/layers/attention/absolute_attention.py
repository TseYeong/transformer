from .attention import BaseAttention


class AbsoluteAttention(BaseAttention):
    """
    Attention with absolute position encoding, without bias.
    """

    def forward(self, x_q, x_kv=None, mask=None):
        if x_kv is None:
            x_kv = x_q

        Q, K, V = self._reshape_qkv(x_q, x_kv)
        context, _ = self.attn(Q, K, V, mask)

        out = context.transpose(1, 2).contiguous().view(x_q.size(0), -1, self.d_model)

        return self.o_proj(out)
