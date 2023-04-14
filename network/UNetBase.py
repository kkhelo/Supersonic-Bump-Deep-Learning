"""
name : UNetBase.py
usage : U-Net architecture, predict surface pressure from bump height, geometry parameters and flow conditions
author : Bo-Yuan You
Date : 2023-03-02

"""

import torch.nn as nn
import torch
from networkBlock import DownSamplingBlock, UpSamplingBlock


class SPUNet(nn.Module):
    """
    Surface Pressure U-Net (SPUNet)
    U-Net architecture use to predict surface pressure from bump heights matrix.
    """
    def __init__(self, inChannel, outChannel, inParaLen : int, channelBase = 64, activation = nn.Tanh()) -> None:
        """
        * inChannel, outChannel : number of channels of input and output layer
        * channelBase number of channels in the first block
        * activation : activation method object to use at the output layer
        """
        super().__init__()

        self.downLayer1 = DownSamplingBlock(inChannel  , channelBase)
        self.downLayer2 = DownSamplingBlock(channelBase  , channelBase*2)
        self.downLayer3 = DownSamplingBlock(channelBase*2, channelBase*2)
        self.downLayer4 = DownSamplingBlock(channelBase*2, channelBase*4)
        self.downLayer5 = DownSamplingBlock(channelBase*4, channelBase*4)
        self.downLayer6 = DownSamplingBlock(channelBase*4, channelBase*8)
        self.downLayer7 = DownSamplingBlock(channelBase*8, channelBase*8)
        self.downLayer8 = DownSamplingBlock(channelBase*8, channelBase*16)

        additionalBottleNeck = []
        additionalBottleNeck.append(nn.Linear(inParaLen, channelBase, bias=False))
        additionalBottleNeck.append(nn.BatchNorm1d(channelBase))
        additionalBottleNeck.append(nn.ReLU(inplace=True))
        self.additionalBottleNeck = nn.Sequential(*additionalBottleNeck)

        maskBottleNeck = []
        maskBottleNeck.append(nn.Linear(channelBase*16+channelBase, channelBase*16, bias=False))
        maskBottleNeck.append(nn.BatchNorm1d(channelBase*16))
        maskBottleNeck.append(nn.ReLU(inplace=True))
        self.maskBottleNeck = nn.Sequential(*maskBottleNeck)
        
        self.upLayer8 = UpSamplingBlock(channelBase*16, channelBase*8)
        self.upLayer7 = UpSamplingBlock(channelBase*16, channelBase*8)
        self.upLayer6 = UpSamplingBlock(channelBase*16, channelBase*4)
        self.upLayer5 = UpSamplingBlock(channelBase*8, channelBase*4)
        self.upLayer4 = UpSamplingBlock(channelBase*8, channelBase*2)
        self.upLayer3 = UpSamplingBlock(channelBase*4, channelBase*2)
        self.upLayer2 = UpSamplingBlock(channelBase*4, channelBase)
        self.upLayer1 = UpSamplingBlock(channelBase*2, outChannel, activation=False)
        self.actf = activation

        # nn.init.kaiming_uniform_(self.additionalBottleNeck[0].weight, mode='fan_in', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.maskBottleNeck[0].weight, mode='fan_in', nonlinearity='relu')

        # if activation == nn.SELU():
        #     nn.init.normal_(self.upLayer1[1].weight, mean=0, std=(1/channelBase*2 * 3 * 3)**0.5)
        # elif activation == nn.Tanh():
        #     nn.init.xavier_uniform_(self.upLayer1[1].weight)
    
    
    def forward(self, inMask, inPara):
        
        # Encoder layers
        downOut1 = self.downLayer1(inMask)
        downOut2 = self.downLayer2(downOut1)
        downOut3 = self.downLayer3(downOut2)
        downOut4 = self.downLayer4(downOut3)
        downOut5 = self.downLayer5(downOut4)
        downOut6 = self.downLayer6(downOut5)
        downOut7 = self.downLayer7(downOut6)
        downOut8 = self.downLayer8(downOut7)

        # Bottleneck
        additionalBottleNeck = self.additionalBottleNeck(inPara)
        maskBottleNeck = self.maskBottleNeck(torch.cat([downOut8.squeeze(2).squeeze(2), additionalBottleNeck], dim=1)).unsqueeze(2).unsqueeze(2)

        # Decoder layer
        upOut8 = self.upLayer8(maskBottleNeck)
        upOut7 = self.upLayer7(torch.cat([upOut8, downOut7],1))
        upOut6 = self.upLayer6(torch.cat([upOut7, downOut6],1))
        upOuT5 = self.upLayer5(torch.cat([upOut6, downOut5],1))
        upOut4 = self.upLayer4(torch.cat([upOuT5, downOut4],1))
        upOut3 = self.upLayer3(torch.cat([upOut4, downOut3],1))
        upOut2 = self.upLayer2(torch.cat([upOut3, downOut2],1))
        upOut1 = self.upLayer1(torch.cat([upOut2, downOut1],1))
      
        return self.actf(upOut1)
