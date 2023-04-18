"""
name : AE.py
usage : Autoencoder architecture
author : Bo-Yuan You
Date : 2023-04-15

"""

import torch.nn as nn
import torch
from .networkBlock import DownsamplingBlock, UpsamplingBlock


class AE(nn.Module):
    """
        Autoencoder with adjustable length and number of channels
    """
    def __init__(self, inChannel, outChannel, channelBase : int, channelFactors : list(),inVectorLength : int,  
                 activation = nn.Tanh(), resolution : int = 256, bias : bool = True) -> None:
        """
            * inChannel, outChannel : Number of channels of input and output layer
            * inVectorlength : Length of vectors input feed in the bottleneck
            * channelBase : Number of channels in the first block
            * channelFactors : List of channel factor, channels in each layer = channelBase * factor at that layer
            * activation : Activation method for the output layer
            * resolution : Input images resolution
            * bias : Whether to add bias term in convolution layer
        """

        super().__init__()
        
        bottlenackResolution = resolution // (2**len(channelFactors))
        print('Resolution at bottle neck in this settings will be : ' ,bottlenackResolution)

        ########## Encoder ##########
        encoder = [DownsamplingBlock(inChannels=inChannel, outChannels=channelBase, bias=bias)]
        for i in range(len(channelFactors)-1): 
            encoder.append(DownsamplingBlock(inChannels=channelBase*channelFactors[i], outChannels=channelBase*channelFactors[i+1], bias=bias))
        # Make sure the size to be (1,1)
        encoder.append(nn.AvgPool2d(kernel_size=bottlenackResolution) if bottlenackResolution != 1 else nn.Identity())

        ########## Bottleneck and VectorBlock ##########

        # VectorBlock for input vector processing
        vectorBlock = []
        vectorBlock.append(nn.Linear(inVectorLength, channelBase, bias=bias))
        vectorBlock.append(nn.BatchNorm1d(channelBase))
        vectorBlock.append(nn.ReLU(inplace=True))

        # Bottleneck
        bottleneck = []
        bottleneckInChannels = channelBase*channelFactors[-1]+channelBase
        bottleneckOutChannels = channelBase*channelFactors[-1]
        bottleneck.append(nn.Linear(bottleneckInChannels, bottleneckOutChannels, bias=bias))
        bottleneck.append(nn.BatchNorm1d(bottleneckOutChannels))
        bottleneck.append(nn.ReLU(inplace=True))

        # VectorBlock and bottleneck weight initializtion
        nn.init.kaiming_uniform_(vectorBlock[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(bottleneck[0].weight, mode='fan_in', nonlinearity='relu')
        
        ########## Decoder ##########

        decoder = [nn.UpsamplingBilinear2d(scale_factor = bottlenackResolution) if bottlenackResolution != 1 else nn.Identity()]
        
        for i in range(1, len(channelFactors)): 
            decoder.append(UpsamplingBlock(inChannels=channelBase*channelFactors[-i], outChannels=channelBase*channelFactors[-i-1], bias=bias))

        decoder.append(UpsamplingBlock(inChannels=channelBase, outChannels=outChannel, activation=activation, bias=False))
        if activation == nn.SELU():
            nn.init.kaiming_uniform_(decoder[-1].conv.weight, mode='fan_in', nonlinearity='linear')
        elif activation == nn.Tanh():
            nn.init.xavier_uniform_(decoder[-1].conv.weight)
        elif activation == nn.ReLU():
            nn.init.kaiming_uniform_(decoder[-1].weight, mode='fan_in', nonlinearity='relu')

        ########## Create sequential object ##########
        self.encoder = nn.Sequential(*encoder)
        self.vectorBlock = nn.Sequential(*vectorBlock)
        self.bottleneck = nn.Sequential(*bottleneck)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, inMap, inVec):
        
        xMap = inMap
        # Encoder
        for block in self.encoder:
            xMap = block(xMap)
        
        size = xMap.size()

        # Bottleneck
        xMap = xMap.view(size[0], -1)
        xVec = self.vectorBlock(inVec)
        xMap = self.bottleneck(torch.cat([xMap, xVec], dim=1))
        xMap = xMap.view(size)

        # Decoder 
        for block in self.decoder:
            xMap = block(xMap)

        return xMap
