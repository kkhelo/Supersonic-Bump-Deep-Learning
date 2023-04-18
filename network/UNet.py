"""
name : UNet.py
usage : U-Net architecture
author : Bo-Yuan You
Date : 2023-03-02

"""

import torch.nn as nn
import torch
from .networkBlock import DownsamplingBlock, UpsamplingBlock


class UNet(nn.Module):
    """
        Unet with adjustable length and number of channels, optional final block feature
    """
    def __init__(self, inChannel, outChannel, channelBase : int, channelFactors : list(), inVectorLength : int, 
                 finalBlockDivisors = None, activation = nn.Tanh(), resolution : int = 256, bias : bool = True) -> None:
        """
            * inChannel, outChannel : Number of channels of input and output layer
            * inVectorlength : Length of vectors input feed in the bottleneck
            * channelBase : Number of channels in the first block
            * channelFactors : List of channel factor, channels in each layer = channelBase * factor at that layer
            * finalBlockDivisors : Append final block at the end, adjustable length and number of channels
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

        # Reshape the feature advance to decode
        
        decoder = [nn.UpsamplingBilinear2d(scale_factor = bottlenackResolution) if bottlenackResolution != 1 else nn.Identity()]

        decoder.append(UpsamplingBlock(inChannels=channelBase*channelFactors[-1], outChannels=channelBase*channelFactors[-2], bias=bias))
        for i in range(2, len(channelFactors)): 
            decoder.append(UpsamplingBlock(inChannels=channelBase*channelFactors[-i]*2, outChannels=channelBase*channelFactors[-i-1], bias=bias))

        ########## FinalBlock ##########
        if finalBlockDivisors:
            decoder.append(UpsamplingBlock(inChannels=channelBase*2, outChannels=channelBase, bias=bias))
            finalBlockChannels = [channelBase] + [channelBase//divisor for divisor in finalBlockDivisors]
            finalBlock = []

            for i in range(len(finalBlockChannels)-1):
                finalBlock.append(nn.Conv2d(finalBlockChannels[i], finalBlockChannels[i+1], kernel_size=3, stride=1, padding=1, bias=bias))
                finalBlock.append(nn.BatchNorm2d(finalBlockChannels[i+1]))
                finalBlock.append(nn.ReLU())
                
                nn.init.kaiming_uniform_(finalBlock[i*3].weight, mode='fan_in', nonlinearity='relu')

            finalBlock.append(nn.Conv2d(finalBlockChannels[-1], outChannel, kernel_size=3, stride=1, padding=1, bias=bias))
            finalBlock.append(nn.BatchNorm2d(outChannel))
            finalBlock.append(activation)

            if activation == nn.SELU():
                nn.init.kaiming_uniform_ (finalBlock[-3].weight, mode='fan_in', nonlinearity='linear')
            elif activation == nn.Tanh():
                nn.init.xavier_uniform_(finalBlock[-3].weight)
            elif activation == nn.ReLU():
                nn.init.kaiming_uniform_(finalBlock[-3].weight, mode='fan_in', nonlinearity='relu')
        else:
            decoder.append(UpsamplingBlock(inChannels=channelBase*2, outChannels=outChannel, activation=activation, bias=bias))
            finalBlock = [nn.Identity()]
            if activation == nn.SELU():
                nn.init.kaiming_uniform_(decoder[-1].conv.weight, mode='fan_in', nonlinearity='linear')
            elif activation == nn.Tanh():
                nn.init.xavier_uniform_(decoder[-1].conv.weight)

        ########## Create sequential object ##########
        self.encoder = nn.Sequential(*encoder)
        self.vectorBlock = nn.Sequential(*vectorBlock)
        self.bottleneck = nn.Sequential(*bottleneck)
        self.decoder = nn.Sequential(*decoder)
        self.finalBlock = nn.Sequential(*finalBlock)

    def forward(self, inMap, inVec):    
        
        xMap = self.encoder[0](inMap)
        encoderFeatures = []

        # Encoder
        for block in self.encoder[1:]:
            encoderFeatures.append(xMap.clone())
            xMap = block(xMap)
            
        size = xMap.size()

        # Bottleneck
        xMap = xMap.view(size[0], -1)
        xVec = self.vectorBlock(inVec)
        xMap = self.bottleneck(torch.cat([xMap, xVec], dim=1))
        xMap = xMap.view(size)

        # Decoder layer
        xMap = self.decoder[0](xMap)
        xMap = self.decoder[1](xMap)
        for i in range(2, len(self.decoder)):
            xMap = self.decoder[i](torch.cat([xMap, encoderFeatures[-i]],1))

        # Final block
        xMap = self.finalBlock(xMap)

        return xMap
