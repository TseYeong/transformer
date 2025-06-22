import torch
from torch import nn
import math
from abc import ABC, abstractmethod


class BasePositionalEncoding(nn.Module, ABC):
    """
    Class of positional encoding
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
    @abstractmethod
    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, seq_len, d_model
        :type x: Tensor
        :return: Same shape tensor, with positional info added
        :rtype: Tensor
        """
        pass


class AbsolutePositionalEncoding(BasePositionalEncoding):
    """
    Implements the sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__(d_model, dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(), :].requires_grad_(False)
        return self.dropout(x)
