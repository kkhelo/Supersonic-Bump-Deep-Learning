"""
name : retriveLossHistory.py
usage : Some self define activated function
author : Bo-Yuan You
Date : 2023-03-10

"""

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys, os

modelName = sys.argv[1]
outputsFileName = sys.argv[2]
eventTrain = EventAccumulator(os.path.join(modelName, 'Loss_Train/'))
eventVal = EventAccumulator(os.path.join(modelName, 'Loss_Validation/'))
eventTrain.Reload()
eventVal.Reload()
lossTrain = [s.value for s in eventTrain.Scalars('Loss')]
lossVal = [s.value for s in eventVal.Scalars('Loss')]

outputsFile = os.path.join(outputsFileName)
np.savez(outputsFile, train=lossTrain, val=lossVal)