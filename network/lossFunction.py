"""
name : LossFunction.py
usage : Some self define loss function
author : Bo-Yuan You
Date : 2023-03-10

"""

import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, weight=1):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        # Compute the gradient of the output image along both horizontal and vertical directions
        dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])

        # Sum up the absolute values of the gradients to get the total variation
        tv = torch.sum(dx) + torch.sum(dy)

        # Multiply the total variation by the weight
        loss = self.weight * tv

        return loss
