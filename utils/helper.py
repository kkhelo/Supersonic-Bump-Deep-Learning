"""
name : helper.py
usage : some helper function
author : Bo-Yuan You
Date : 2023-03-10

"""

def retriveLossHistory(casePath, outputsFileName):
    """
        Convert tensorboard log summary log file into numpy file

        Args:
        - casePath (str): Path to the directory where the loss is.
        - outputsFileName (str): Path where to store loss history.
    """
    import numpy as np, os
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    eventTrain = EventAccumulator(os.path.join(casePath, 'L1_Train/'))
    eventVal = EventAccumulator(os.path.join(casePath, 'L1_Validation/'))
    eventTrain.Reload()
    eventVal.Reload()
    lossTrain = [s.value for s in eventTrain.Scalars('L1')]
    lossVal = [s.value for s in eventVal.Scalars('L1')]

    outputsFile = os.path.join(outputsFileName)
    np.savez(outputsFile, train=lossTrain, val=lossVal)

def trainingRecord(params, filename):
    """
        Records hyperparameters used in training to a JSON file.
        
        Args:
        - params (dict): A dictionary of hyperparameters and their values.
        - filename (str): The name of the file to save the hyperparameters to.
    """
    import json
    with open(filename, 'w') as of:
        json.dump(params, of, indent=4)