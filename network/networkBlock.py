"""
name : networkBlock.py
usage : U-Net block component implementation.
author : Bo-Yuan You
Date : 2023-03-02
"""

import torch.nn as nn
import torch


class DownsamplingBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation = nn.ReLU(inplace=True), bias = True) -> None:
        """
            * inChannel, outChannel : Number of channels of input and output
            * activation : Activation module, default is ReLU
            * bias : Whether to activate bias term
        """
        super().__init__()

        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(outChannels)
        self.actf = activation

        if isinstance(activation, nn.ReLU) : nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.actf(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, inChannels, outChannels, activation = nn.ReLU(inplace=True), bias = True) -> None:
        """
            * inChannel, outChannel : Number of channels of input and output
            * activation : Activation module, default is ReLU
            * bias : Whether to activate bias term
        """
        super().__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn = nn.BatchNorm2d(outChannels)
        self.actf = activation

        if isinstance(activation, nn.ReLU) : nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.actf(x)


class PAM(nn.Module):
    """ 
        Position attention module (PAM)
    """
    def __init__(self, inChannels, bias = True):
        super().__init__()
        self.inChannels = inChannels

        self.queryConv = nn.Conv2d(in_channels=inChannels, out_channels=inChannels//8, kernel_size=1, bias=bias)
        self.keyConv = nn.Conv2d(in_channels=inChannels, out_channels=inChannels//8, kernel_size=1, bias=bias)
        self.valueConv = nn.Conv2d(in_channels=inChannels, out_channels=inChannels, kernel_size=1, bias=bias)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)                                                                                                                                                           

        nn.init.xavier_uniform_(self.queryConv.weight)
        nn.init.xavier_uniform_(self.keyConv.weight)
        nn.init.xavier_uniform_(self.valueConv.weight)

    def forward(self, x):
        """
            Args :
                x : input feature maps(batch, channels, height, width)
            returns :
                out : attention value + input feature
                attention: batchSize X (HxW) X (HxW)
        """
        batchSize, channels, height, width = x.size()
        
        # quert path
        matQuery = self.queryConv(x).view(batchSize, -1, width*height).permute(0, 2, 1)

        # key path
        matkey = self.keyConv(x).view(batchSize, -1, width*height)

        # attention
        energy = torch.bmm(matQuery, matkey)
        attention = self.softmax(energy)

        # value path
        matValue = self.valueConv(x).view(batchSize, -1, width*height)

        # attention x value
        out = torch.bmm(matValue, attention.permute(0, 2, 1)).view(batchSize, channels, height, width)

        # residual connection
        out = self.gamma*out + x

        return out


class CAM(nn.Module):
    """ 
        Channel attention module (CAM)
    """
    def __init__(self, inChannel):
        super().__init__()
        self.inChannel = inChannel

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
            Args :
                x : input feature maps(batch, channels, H, W)
            returns :
                out : attention value + input feature
                attention: (batchSize, channels, channels)
        """
        batchSize, channels, height, width = x.size()
        
        # quert path
        matQuery = x.view(batchSize, channels, -1)

        # key path
        matKey = x.view(batchSize, channels, -1).permute(0, 2, 1)

        # attention
        energy = torch.bmm(matQuery, matKey)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)

        # value path
        matValue = x.view(batchSize, channels, -1)

        # attention x value
        out = torch.bmm(attention, matValue).view(batchSize, channels, height, width)

        # residual
        out = self.gamma*out + x
        
        return out