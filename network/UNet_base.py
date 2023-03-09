"""
name : UNet_base.py
usage : U-Net architecture, predict surface pressure from bump height, geometry parameters and flow conditions
author : Bo-Yuan You
Date : 2023-03-02

"""

import torch 
import torch.nn as nn


class DownSamplingblock(nn.Module):
    def __init__(self, in_channel, out_channel, bn=False, relu=0, size=4, pad=1) -> None:
        super().__init__()

        net = []
        net.append(nn.ReLU(inplace=True) if not relu else nn.LeakyReLU(relu, inplace=True))
        net.append(nn.Conv2d(in_channel, out_channel, kernel_size=size, stride=2, padding=pad, bias=True))
        if bn : net.append(nn.BatchNorm2d(out_channel))
        
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bn=False, relu=0, size=4, pad=1) -> None:
        super().__init__()

        net = []
        net.append(nn.ReLU(inplace=True) if not relu else nn.LeakyReLU(relu, inplace=True))
        net.append(nn.UpsamplingBilinear2d(scale_factor=2))
        net.append(nn.Conv2d(in_channel, out_channel, kernel_size=size-1, stride=1, padding=pad, bias=True))
        if bn : net.append(nn.BatchNorm2d(out_channel))
        
        self.net = nn.Sequential(*net)


    def forward(self, x):
        out = self.net(x)
        return out


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, inParaLen : int, expo = 6, reluAlpha : float = 0.2) -> None:
        super().__init__()

        channel = 2 ** int(expo)
        self.downLayer1 = nn.Sequential(nn.Conv2d(in_channel, channel, kernel_size=4, stride=2, padding=1, bias=True))
        self.downLayer2 = DownSamplingblock(channel  , channel*2, bn=True , relu=reluAlpha)
        self.downLayer3 = DownSamplingblock(channel*2, channel*2, bn=True , relu=reluAlpha)
        self.downLayer4 = DownSamplingblock(channel*2, channel*4, bn=True , relu=reluAlpha)
        self.downLayer5 = DownSamplingblock(channel*4, channel*4, bn=True , relu=reluAlpha)
        self.downLayer6 = DownSamplingblock(channel*4, channel*8, bn=True , relu=reluAlpha, size=2, pad=0)
        self.downLayer7 = DownSamplingblock(channel*8, channel*8, bn=False, relu=reluAlpha, size=2, pad=0)
        self.downLayer8 = DownSamplingblock(channel*8, channel*16, bn=False, relu=reluAlpha, size=2, pad=0)

        self.bottleNeck1 = nn.Linear(channel*16, channel*16)
        self.bottleNeck2 = nn.Linear(channel*16+inParaLen, channel*16)

        self.upLayer8 = UpSamplingBlock(channel*16 , channel*8, bn=True, relu=0, size=2, pad=0)
        self.upLayer7 = UpSamplingBlock(channel*16 , channel*8, bn=True, relu=0, size=2, pad=0)
        self.upLayer6 = UpSamplingBlock(channel*16, channel*4, bn=True, relu=0, size=2, pad=0)
        self.upLayer5 = UpSamplingBlock(channel*8, channel*4, bn=True, relu=0)
        self.upLayer4 = UpSamplingBlock(channel*8, channel*2, bn=True, relu=0)
        self.upLayer3 = UpSamplingBlock(channel*4, channel*2, bn=True, relu=0)
        self.upLayer2 = UpSamplingBlock(channel*4, channel  , bn=True, relu=0)
        
        upLayer1 = []
        upLayer1.append(nn.ReLU(inplace=True))
        upLayer1.append(nn.UpsamplingBilinear2d(scale_factor=2))
        upLayer1.append(nn.Conv2d(channel*2, channel, kernel_size=3, stride=1, padding=1, bias=True))
        self.upLayer1 = nn.Sequential(*upLayer1)

        finalLayer = []
        finalLayer.append(nn.Conv2d(channel, channel//2, kernel_size=3, stride=1, padding=1, bias=True))
        finalLayer.append(nn.Conv2d(channel//2, channel//2, kernel_size=3, stride=1, padding=1, bias=True))
        finalLayer.append(nn.Conv2d(channel//2, out_channel, kernel_size=3, stride=1, padding=1, bias=True))
        self.finalLayer = nn.Sequential(*finalLayer)

        # upLayer1 = []
        # upLayer1.append(nn.ReLU(inplace=True))
        # upLayer1.append(nn.ConvTranspose2d(channel*2, out_channel, kernel_size=4, stride=2, padding=1, bias=True))

        

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
        bottleNeck1 = self.bottleNeck1(torch.flatten(downOut8,start_dim=1))
        bottleNeck2 = self.bottleNeck2(torch.cat([bottleNeck1, inPara], dim=1)).unsqueeze(2).unsqueeze(2)

        # Decoder layer
        upOut8 = self.upLayer8(bottleNeck2)
        upOut7 = self.upLayer7(torch.cat([upOut8, downOut7],1))
        upOut6 = self.upLayer6(torch.cat([upOut7, downOut6],1))
        upOuT5 = self.upLayer5(torch.cat([upOut6, downOut5],1))
        upOut4 = self.upLayer4(torch.cat([upOuT5, downOut4],1))
        upOut3 = self.upLayer3(torch.cat([upOut4, downOut3],1))
        upOut2 = self.upLayer2(torch.cat([upOut3, downOut2],1))
        upOut1 = self.upLayer1(torch.cat([upOut2, downOut1],1))
        final  = self.finalLayer(upOut1)
        
        return final