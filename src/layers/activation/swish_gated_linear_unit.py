from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.
    Combines the Swish activation with Gated Linear Units.
    """

    def __init__(self, d_model, hidden_dim=None):
        super(SwiGLU, self).__init__()
        if hidden_dim is None:
            hidden_dim = d_model * 4

        self.fc1 = nn.Linear(d_model, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x_proj = self.fc1(x)
        x_1, x_2 = x_proj.chunk(2, dim=-1)
        out = self.fc2(x_1 * F.silu(x_2))
        return out