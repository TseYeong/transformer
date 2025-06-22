import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLayerNorm(nn.Module, ABC):
    def __init__(self):
        super(BaseLayerNorm, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass
