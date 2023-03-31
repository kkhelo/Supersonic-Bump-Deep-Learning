"""
name : ActivatedFunction.py
usage : Some self define activated function
author : Bo-Yuan You
Date : 2023-03-10

"""

import torch 
import torch.nn as nn


class Swish1(nn.Module):
    """
    f(x) = x * (1 + exp(-beta * x))
    """
    def __init__(self, beta=1.0):
        super(Swish1, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
class Swish(nn.Module):
    """
    f(x) = x * (1 + exp(-beta * x))
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)