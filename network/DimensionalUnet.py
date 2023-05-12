"""
name : DimensionalUnet.py
usage : DimensionalUnet architecture
author : Bo-Yuan You
Date : 2023-04-21

"""

import torch.nn as nn
import torch
from .networkBlock import UpsamplingBlock, DownsamplingBlock
    
    
class DimensionalUnet(nn.Module):
    """
        DimensionalUnet with adjustable length and number of channels
        Global encoder has same channels factors as local encoder.
    """
    def __init__(self, outChannel, channelBase: int, channelFactors: list, inVectorLength: int, globalEncoder: nn.Module, 
                 bottleneck: nn.Module, activation=nn.Tanh(), resolution: int = 256, bias: bool = True) -> None:
        """
            * inChannel, outChannel : Number of channels of input and output layer
            * channelBase : Number of channels at first block of local encoder
            * channelFactors : List of channel factor, channels in each layer = channelBase * factor at that layer
            * inVectorlength : Length of vectors input feed in the bottleneck
            * converter : Converter module convert 3D bump information into 2D slice information
            * activation : Activation method for the output layer
            * resolution : Input images resolution
            * bias : Whether to add bias term in convolution layer
        """
        super().__init__()

        bottlenackResolution = resolution // (2**len(channelFactors))

        ########## Loacl encoder ##########

        localEncoder = [DownsamplingBlock(inChannels=1, outChannels=channelBase, bias=bias)]
        for i in range(len(channelFactors)-1): 
            localEncoder.append(DownsamplingBlock(inChannels=channelBase*channelFactors[i], outChannels=channelBase*channelFactors[i+1], bias=bias))
        
        ########## VectorBlock ##########

        # VectorBlock for input vector processing
        vectorBlock = []
        vectorBlock.append(nn.Linear(inVectorLength, channelBase, bias=bias))
        vectorBlock.append(nn.BatchNorm1d(channelBase))
        vectorBlock.append(nn.ReLU(inplace=True))
        
        # VectorBlock and bottleneck weight initializtion
        nn.init.kaiming_uniform_(vectorBlock[0].weight, mode='fan_in', nonlinearity='relu')

        ########## Decoder ##########

        # Reshape features
        decoder = [nn.UpsamplingBilinear2d(scale_factor = bottlenackResolution) if bottlenackResolution != 1 else nn.Identity()]

        for i in range(1, len(channelFactors)): 
            decoder.append(UpsamplingBlock(inChannels=channelBase*channelFactors[-i]*2, outChannels=channelBase*channelFactors[-i-1], bias=bias))

        decoder.append(UpsamplingBlock(inChannels=channelBase*2, outChannels=outChannel, activation=activation, bias=bias))
        if activation == nn.SELU():
            nn.init.kaiming_uniform_(decoder[-1].conv.weight, mode='fan_in', nonlinearity='linear')
        elif activation == nn.Tanh():
            nn.init.xavier_uniform_(decoder[-1].conv.weight)

        ########## Create sequential object ##########

        self.localEncoder = nn.Sequential(*localEncoder)
        self.globalEncoder = globalEncoder
        self.vectorBlock = nn.Sequential(*vectorBlock)
        self.bottleneck = bottleneck
        self.decoder = nn.Sequential(*decoder)

    def forward(self, inMap, inVec, inBinaryMask):    

        # Global Encoder
        xMapGlobal = self.globalEncoder(inMap)

        # Local Encoder
        encoderFeatures = []
        xMapLocal = inBinaryMask.clone()

        for block in self.localEncoder:
            xMapLocal = block(xMapLocal)
            encoderFeatures.append(xMapLocal.clone())
            
        # Bottleneck and vectorBlock
        xVec = self.vectorBlock(inVec)
        xMap = self.bottleneck(xMapGlobal, xVec)

        # Decoder layer
        xMap = self.decoder[0](xMap)

        for i in range(1, len(self.decoder)):
            xMap = self.decoder[i](torch.cat([xMap, encoderFeatures[-i]], dim=1))

        return xMap
        