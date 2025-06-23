from torch import nn
from .activation.swish_gated_linear_unit import SwiGLU


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1, activate='relu'):
        super(PointWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

        ACT_FN = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "swiglu": SwiGLU(hidden)
        }
        assert activate in ACT_FN

        self.act = ACT_FN[activate]
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        out = self.fc2(x)

        return out
