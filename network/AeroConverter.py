"""
name : AeroConverter.py
usage : AeroConverter and its submodules architecture
author : Bo-Yuan You
Date : 2023-03-02

"""

import torch.nn as nn
import torch
from .networkBlock import DownsamplingBlock, UpsamplingBlock
from .AE import AE


class AttentionConverter(AE):
    """
        AttentionConverter module extend from AE, add attention modules
    """


class GlobalEncoderCNN(nn.Module):
    """
        GlobalEncoderCNN with adjustable length and number of channels. This module is specifically designed for AeroConverter
    """
    def __init__(self, inChannel, channelBase : int, channelFactors : list, 
                 resolution : int = 256, bias : bool = True) -> None:
        super().__init__()

        bottlenackResolution = resolution // (2**len(channelFactors))

        net = [DownsamplingBlock(inChannels=inChannel, outChannels=channelBase, bias=bias)]
        for i in range(len(channelFactors)-1): 
            net.append(DownsamplingBlock(inChannels=channelBase*channelFactors[i], outChannels=channelBase*channelFactors[i+1], bias=bias))
        
        # Make sure the size to be (1,1)
        net.append(nn.AvgPool2d(kernel_size=bottlenackResolution) if bottlenackResolution != 1 else nn.Identity())

        self.net = nn.Sequential(*net)

    def forward(self, inMap):    
        
        xMap = inMap.clone()

        for block in self.net : xMap = block(xMap)
            
        return xMap


class AeroConverter(nn.Module):
    """
        AeroConverter with adjustable length and number of channels
        Global encoder has same channels factors as local encoder
    """
    def __init__(self, outChannel, channelBase : int, channelFactors : list, inVectorLength : int, converter : nn.Module, 
                 globalEncoder : nn.Module, activation = nn.Tanh(), resolution : int = 256, bias : bool = True) -> None:
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

        # Reshape the feature advance to decode
        decoder = [nn.UpsamplingBilinear2d(scale_factor = bottlenackResolution) if bottlenackResolution != 1 else nn.Identity()]

        decoder.append(UpsamplingBlock(inChannels=channelBase*channelFactors[-1], outChannels=channelBase*channelFactors[-2], bias=bias))
        for i in range(2, len(channelFactors)): 
            decoder.append(UpsamplingBlock(inChannels=channelBase*channelFactors[-i]*2, outChannels=channelBase*channelFactors[-i-1], bias=bias))

        decoder.append(UpsamplingBlock(inChannels=channelBase*2, outChannels=outChannel, activation=activation, bias=bias))
        if activation == nn.SELU():
            nn.init.kaiming_uniform_(decoder[-1].conv.weight, mode='fan_in', nonlinearity='linear')
        elif activation == nn.Tanh():
            nn.init.xavier_uniform_(decoder[-1].conv.weight)

        ########## Create sequential object ##########

        self.converter = converter
        self.localEncoder = nn.Sequential(*localEncoder)
        self.globalEncoder = globalEncoder
        self.vectorBlock = nn.Sequential(*vectorBlock)
        self.bottleneck = nn.Sequential(*bottleneck)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, inMap, inVec):    
        
        # Converter (Only pass last 4 parameters into converter)
        outConverter = self.converter(inMap, inVec[1:])
        xMapLocal = (nn.Sigmoid(outConverter) > 0.5).float()

        # Global Encoder
        xMapGlobal = self.globalEncoder(inMap)

        # Local Encoder
        xMapLocal = self.localEncoder[0](xMapLocal)
        encoderFeatures = []

        for block in self.localEncoder[1:]:
            encoderFeatures.append(xMapLocal.clone())
            xMapLocal = block(xMapLocal)
            
        size = xMapLocal.size()

        # Bottleneck
        xMapGlobal = xMapGlobal.view(size[0], -1)
        xVec = self.vectorBlock(inVec)
        xMapGlobal = self.bottleneck(torch.cat([xMapGlobal, xVec], dim=1))
        xMapGlobal = xMapGlobal.view(size)

        # Decoder layer
        xMapDecoder = xMapGlobal.clone()

        xMapDecoder = self.decoder[0](xMapDecoder)
        xMapDecoder = self.decoder[1](xMapDecoder)
        for i in range(2, len(self.decoder)):
            xMapDecoder = self.decoder[i](torch.cat([xMapDecoder, encoderFeatures[-i]],1))

        return xMapDecoder, outConverter
