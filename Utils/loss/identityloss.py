import torch
from torch import nn as nn

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loss, label=None):
        if isinstance(loss, torch.Tensor):
            if len(loss.shape) == 0:
                return loss
            else:
                return loss.mean()
        elif isinstance(loss, list):
            return sum(l for l in loss)
        elif isinstance(loss, tuple):
            return sum(l for l in loss)
        elif isinstance(loss, dict):
            return sum(l for l in loss.values())
        else:
            return loss