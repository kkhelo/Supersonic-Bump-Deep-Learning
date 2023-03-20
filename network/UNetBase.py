"""
name : UNet_base.py
usage : U-Net architecture, predict surface pressure from bump height, geometry parameters and flow conditions
author : Bo-Yuan You
Date : 2023-03-02

"""

import torch.nn as nn
import torch


class DownSamplingblock(nn.Module):
    def __init__(self, in_channel, out_channel, activation = nn.ReLU(inplace=True)) -> None:
        super().__init__()

        net = []
        net.append(nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=True))
        net.append(nn.BatchNorm2d(out_channel))
        net.append(activation)
        
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activation = nn.ReLU(inplace=True)) -> None:
        super().__init__()

        net = []
        net.append(nn.UpsamplingBilinear2d(scale_factor=2))
        net.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True))
        net.append(nn.BatchNorm2d(out_channel))
        net.append(activation)
        
        self.net = nn.Sequential(*net)


    def forward(self, x):
        out = self.net(x)
        return out


class SPUNet(nn.Module):
    """
    Surface Pressure U-Net (SPUNet)
    U-Net architecture use to predict surface pressure from bump heights matrix.
    """
    def __init__(self, in_channel, out_channel, inParaLen : int, expo = 6, activation = nn.ReLU(inplace=True)) -> None:
        """
        * expo : channel expand exponent, number of channels after first layer will be 2**expo
        * reluAlpha : parameters to blend from ReLU (reluAlpha=0) to leaky ReLU(0 < reluAlpha < 1)
        """
        super().__init__()

        channel = 2 ** int(expo)
        self.downLayer1 = nn.Sequential(nn.Conv2d(in_channel, channel, kernel_size=4, stride=2, padding=1, bias=True))
        self.downLayer2 = DownSamplingblock(channel  , channel*2, activation)
        self.downLayer3 = DownSamplingblock(channel*2, channel*2, activation)
        self.downLayer4 = DownSamplingblock(channel*2, channel*4, activation)
        self.downLayer5 = DownSamplingblock(channel*4, channel*4, activation)
        self.downLayer6 = DownSamplingblock(channel*4, channel*8, activation)
        self.downLayer7 = DownSamplingblock(channel*8, channel*8, activation)
        self.downLayer8 = DownSamplingblock(channel*8, channel*16, activation)

        # self.bottleNeck1 = nn.Linear(channel*16, channel*16)
        # self.bottleNeck2 = nn.Linear(channel*16+inParaLen, channel*16)

        additionalBottleNeck = []
        additionalBottleNeck.append(nn.Linear(inParaLen, channel))
        additionalBottleNeck.append(nn.BatchNorm1d(channel))
        self.additionalBottleNeck = nn.Sequential(*additionalBottleNeck)

        maskBottleNeck = []
        maskBottleNeck.append(nn.Linear(channel*16+channel, channel*16))
        maskBottleNeck.append(nn.BatchNorm1d(channel*16))
        maskBottleNeck.append(activation)
        self.maskBottleNeck = nn.Sequential(*maskBottleNeck)
        

        self.upLayer8 = UpSamplingBlock(channel*16, channel*8, activation)
        self.upLayer7 = UpSamplingBlock(channel*16, channel*8, activation)
        self.upLayer6 = UpSamplingBlock(channel*16, channel*4, activation)
        self.upLayer5 = UpSamplingBlock(channel*8, channel*4, activation)
        self.upLayer4 = UpSamplingBlock(channel*8, channel*2, activation)
        self.upLayer3 = UpSamplingBlock(channel*4, channel*2, activation)
        self.upLayer2 = UpSamplingBlock(channel*4, channel, activation)
        self.upLayer1 = UpSamplingBlock(channel*2, channel, activation)
        
        # upLayer1 = []
        # upLayer1.append(activation)
        # upLayer1.append(nn.UpsamplingBilinear2d(scale_factor=2))
        # upLayer1.append(nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1, bias=True))
        # self.upLayer1 = nn.Sequential(*upLayer1)

        finalLayer = []
        finalLayer.append(nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1, bias=True))
        finalLayer.append(nn.BatchNorm2d(channel//2))
        finalLayer.append(nn.Conv2d(channel//2, channel//2, kernel_size=3, stride=1, padding=1, bias=True))
        finalLayer.append(nn.BatchNorm2d(channel//2))
        finalLayer.append(nn.Conv2d(channel//2, out_channel, kernel_size=3, stride=1, padding=1, bias=True))
        finalLayer.append(nn.BatchNorm2d(out_channel))
        finalLayer.append(activation)
        self.finalLayer = nn.Sequential(*finalLayer)


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

        # Bottle
        # bottleNeck1 = self.bottleNeck1(torch.flatten(downOut8,start_dim=1))
        # bottleNeck2 = self.bottleNeck2(torch.cat([bottleNeck1, inPara], dim=1)).unsqueeze(2).unsqueeze(2)        
        # additionalBottleNeck = self.additionalBottleNeck(inPara)
        additionalBottleNeck = self.additionalBottleNeck(inPara)
        downOut8 = downOut8.squeeze(2).squeeze(2)
        maskBottleNeck = self.maskBottleNeck(torch.cat([downOut8, additionalBottleNeck], dim=1)).unsqueeze(2).unsqueeze(2)

        # Decoder layer
        # upOut8 = self.upLayer8(bottleNeck2)
        upOut8 = self.upLayer8(maskBottleNeck)
        upOut7 = self.upLayer7(torch.cat([upOut8, downOut7],1))
        upOut6 = self.upLayer6(torch.cat([upOut7, downOut6],1))
        upOuT5 = self.upLayer5(torch.cat([upOut6, downOut5],1))
        upOut4 = self.upLayer4(torch.cat([upOuT5, downOut4],1))
        upOut3 = self.upLayer3(torch.cat([upOut4, downOut3],1))
        upOut2 = self.upLayer2(torch.cat([upOut3, downOut2],1))
        upOut1 = self.upLayer1(torch.cat([upOut2, downOut1],1))
        final  = self.finalLayer(upOut1)
        
        return final
    

