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
        super().__init__()
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
    

class PIContinuityLoss(nn.Module):
    def __init__(self, wallValue, mean : bool = True) -> None:
        """
            * wallValue : wall value (typically 0) after normalization. 
                        This code will automatically calculate neccesary part.
            * mean : mean loss or absolute loss
        """
        super().__init__()
        self.wallValue = torch.tensor([wallValue[2]*wallValue[4], wallValue[2]*wallValue[5]])
        self.mean = mean

    def forward(self, prediction, ground, binaryMask):
        """
            Continuity Loss is defined as : Loss = d(rho*u)/dx + d(rho*v)/dy + d(rho*w)/dz.
            Prediction result cannot calculate d(rho*u)/dx, so this method assumes ground truth 
            satisfies continuity equation in tensor form, the d(rho*u)/dx here can be used in 
            prediction as well. 

            wallValue is the pixel value in wall(bump) region, which was 0 before normalization. 
            wallValue = -tarOffset/tarNorm

            ** SOP 
                1. Assign pixel value at bump region to wallValue.
                2. Multiply rho with v, w( recorded as 'rhoUGround' and 'rhoUPrediction'.)
                3. Calculate d(rhoUGround)/dy + d(rhoUGround)/dz( recorded as constContinuity).
                4. Calculate Loss = d(rhoUPrediction)/dy + d(rhoUPrediction)/dz - constContinuity
        """

        predictionCopy, groundCopy = prediction.clone()[:,2:], ground.clone()[:,2:]

        rhoUGround = torch.cat([(groundCopy[:,0]*groundCopy[:,2]).unsqueeze(1), 
                                    (groundCopy[:,0]*groundCopy[:,3]).unsqueeze(1)],dim=1)
        rhoUPrediction = torch.cat([(predictionCopy[:,0]*predictionCopy[:,2]).unsqueeze(1), 
                                (predictionCopy[:,0]*predictionCopy[:,3]).unsqueeze(1)],dim=1)

        for i in range(2):
            rhoUPrediction[:,i] = rhoUPrediction[:,i]*binaryMask[:,0] + self.wallValue[i]*(1-binaryMask[:,0])
            rhoUGround[:,i] = rhoUGround[:,i]*binaryMask[:,0] + self.wallValue[i]*(1-binaryMask[:,0])

        pad = torch.nn.ReplicationPad2d(1)
        rhoUPredictionPad, rhoUGroundPad = pad(rhoUPrediction), pad(rhoUGround)

        # y direction (physical corrdinates, 3rd dimension in tensor) difference
        diffMapPrediction = rhoUPredictionPad[:,0:1,2:,1:-1] + rhoUPredictionPad[:,0:1,:-2,1:-1] - 2*rhoUPrediction
        diffMapGround = rhoUGroundPad[:,0:1,2:,1:-1] + rhoUGroundPad[:,0:1,:-2,1:-1] - 2*rhoUGround

        # z direction (physical corrdinates, 4th dimension in tensor) difference
        diffMapPrediction += rhoUPredictionPad[:,1:,1:-1,2:] + rhoUPredictionPad[:,1:,1:-1,:-2] - 2*rhoUPrediction
        diffMapGround += rhoUGroundPad[:,1:,1:-1,2:] + rhoUGroundPad[:,1:,1:-1,:-2] - 2*rhoUGround

        loss = torch.sum(torch.abs(diffMapPrediction - diffMapGround))
        if self.mean : 
            loss /= diffMapPrediction.numel()

        return abs(loss)
            

if __name__== '__main__':
    ground = torch.rand(16,6,256,256)*2-1
    pred = torch.rand(16,6,256,256)*2-1
    binaryMask = torch.randint(0,2,(16,1,256,256))
    # binaryMask = torch.randint(0,2,(16,256,256))
    import numpy as np
    wallValue = np.ones((6,1))*-1
    # wallValue = torch.tensor(wallValue)

    criterion = PIContinuityLoss(wallValue, mean=True)

    print(type(criterion.__class__.__name__))

