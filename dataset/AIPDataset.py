"""
name : dataset.py
usage : derive dataset class for AIP prediction implementation
author : Bo-Yuan You
Date : 2023-03-20

"""

import numpy as np
import glob, os, time
from torch.utils.data import Dataset
from baseDataset import baseDataset

class AIPDataset(baseDataset):
    def __init__(self, dataDir: str, mode='TRAIN', caseList=None, res=256, ratio: float = 0.8) -> None:
        super().__init__(dataDir, mode, caseList, res, ratio)

if __name__ == '__main__':
    # Dataset directory.
    dataDir = f'data/trainingData1'
    # Dataset usage mode, train or test.
    mode = 'train'
    caseList='data/trainingData1/caseList1.npz'
    # Dataset and the train loader declaration.
    a = AIPDataset(dataDir=dataDir, mode=mode, caseList=caseList)
    print(a.modeUsage)