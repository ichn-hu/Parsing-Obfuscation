import torch
from torch import nn
from torch.nn import functional as F


def hloss(x):
    x = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
    return x.sum(dim=-1)
