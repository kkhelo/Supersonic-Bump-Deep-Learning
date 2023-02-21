"""
name : dataset.py
usage : dataset class implementation
author : Bo-Yuan You
Date : 2023-02-21

"""

import numpy as np
import glob, os
from torch.utils.data import Dataset


class baseDataset(Dataset):
    """
    Base class for bump simulation data, include data processing.
    This base class provide : 
        1. basic preprocessing method
        2. call function SOP 

    Return data in sequence:
        1. inputsMask : bump surface heights matrix
        2. inputsPara : geometry parameters and flow condition in (k, c, d, Mach) order
        3. targets : surface pressure matrix

    ** Dimensionless and offset removal can not apply to the dataset together. ** 
    Args:
        dataDir : Directory where the dataset is, ex: 'dataset/train'.
        dataChannel : (number of input data channels, number of target data channels).
        preprocessingMode : Choose the preprocessing method, offset removal or dimensionless.
        mode : Taining dataset or testing(evaluation) dataset.
    """

    modeUsage = ('TRAIN', 'TEST', 'DEMO')
    modePreProcessing = ('OFFSETREMOVAL', 'DIMENSIONLESS')

    def __init__(self, dataDir:str, preprocessingMode='OFFSETREMOVAL', mode='TRAIN') -> None:
        super().__init__()
        self.__baseDataDir = dataDir
        self.__dataDir = dataDir
        self.__usageMode = mode.upper()
        self.__preprocessingMode = preprocessingMode.upper()

        if self.__usageMode not in self.modeUsage: 
            raise ValueError(F'Invalid usage mode : {self.__usageMode}, Available Options are (TRAIN, TEST, DEMO)')

        if self.__usageMode not in self.modeUsage: 
            raise ValueError(F'Invalid preprocessing mode : {self.__preprocessingMode}, Available Options are (OFFSETREMOVAL, DIMENSIONLESS)')

    def __call__(self, resolution):
        self.loadData(resolution)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        return self.inputsMask[index], self.inputsPara[index], self.targets[index]

    @property
    def baseDataDir(self):
        return self.__baseDataDir

    @property
    def dataDir(self):
        return self.__dataDir

    @property
    def mode(self):
        return self.__usageMode

    @property
    def preprocessingMode(self):
        return self.__preprocessingMode

    def loadData(self, resolution):
        
        geometry = glob.glob(os.path.join(self.__baseDataDir, '*'))
        caseList = []
        for geo in geometry :
            caseList += glob.glob(os.path.join(geo, '*'))

        # Check if resolution is correct or not
        src = os.path.join(caseList[0], os.listdir(caseList[0])[0], 'bumpSurfaceData.npz')
        temp = np.load(src)['heights']
        if temp.shape[0] - resolution:
            raise ValueError(f"Resolution doesn't math\t\n CFD data : {temp.shape[0]}\t\n Given value : {resolution}")

        self.__length = len(caseList)
        self.inputMask = np.zeros((self.__length, resolution, resolution))
        self.targets = np.zeros((self.__length, resolution, resolution))
        self.inputPara = np.zeros((self.__length, 4))

        for i, case in enumerate(caseList):
            print(i, case)
            src = os.path.join(case, os.listdir(case)[0], 'bumpSurfaceData.npz')
            temp = np.load(src)
            self.inputMask[i] = temp['heights']
            self.targets[i] = temp['pressure']

            # Extract geometry parameters and flow conditions
            temp = case.split('/')
            Mach = float(temp[-1])
            temp = temp[-2].split('_')
            k, c, d = float(temp[0].split('k')[-1]), float(temp[1].split('c')[-1]), float(temp[2].split('d')[-1])
            self.inputPara[i] = np.array([k, c, d, Mach])

class childClass(baseDataset):


    def __init__(self, dataDir: str, preprocessingMode='OFFSETREMOVAL', mode='TRAIN') -> None:
        super().__init__(dataDir, preprocessingMode, mode)

    def loadData(self, resolution):
        print('childClass loadData')


if __name__ == '__main__':
    # a = childClass('data/demoData224')
    a = baseDataset('data/demoData256POINT')
    a(resolution=256)
    # print(a.mode)
    