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

    def __init__(self, dataDir:str, mode='TRAIN', caseList = None, res = 256, ratio = 0.8) -> None:
        super().__init__()
        self.baseDataDir = dataDir
        self.dataDir = dataDir
        self.caseList = caseList
        self.resolution = res
        self.ratio = ratio
        self.mode = mode.upper()

        if self.mode not in self.modeUsage: 
            raise ValueError(F'Invalid usage mode : {self.mode}, Available Options are (TRAIN, TEST, DEMO)')

    def __call__(self):
        self.loadData()

        self._getMean()
        if self.mode == self.modeUsage[0]:
            self._removeOffset()

        self._getNormFactor()
        if self.mode == self.modeUsage[0]:
            self._normalization()

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        return self.inputsMask[index], self.inputsPara[index], self.targets[index]

    def _getCaseList(self) -> None:
        # Use given caseList
        if self.caseList : 
            try :
                caseList = np.load(self.caseList)
                self.__traList = caseList['tra']
                self.__valList = caseList['val']
                self.ratio = caseList['ratio']
                return
            except:
                if self.caseList is list : 
                    self.__traList, self.__valList = self.caseList
                raise ValueError(f'Unkown case list {self.caseList}, should be path to the list file or list(file path).')

        geometry = glob.glob(os.path.join(self.baseDataDir, '*'))
        caseList = []
        for geo in geometry :
            caseList += glob.glob(os.path.join(geo, '*'))

        length = len(caseList)
        np.random.shuffle(caseList)
        sepPoint = int(length*self.ratio)

        self.__traList = caseList[:sepPoint+1]
        self.__valList = caseList[sepPoint+1:]
        self.__length = len(self.__traList)
        ratio = np.array([self.ratio])

        # Save fileList with non-repeat name to base directory
        count = 1
        temp = os.path.join(self.baseDataDir, 'caseList1.npz')
        while os.path.exists(temp):
            temp.replace(f'List{count}', f'List1{count+1}')
            count += 1

        np.savez_compressed(temp, tra=self.__traList, val=self.__valList, ratio=ratio)

    def loadData(self):

        self._getCaseList()

        # Check if resolution is correct or not
        src = os.path.join(self.__traList[0], os.listdir(self.__traList[0])[0], 'bumpSurfaceData.npz')
        temp = np.load(src)['heights']
        if temp.shape[0] - self.resolution:
            raise ValueError(f"Resolution doesn't math\t\n CFD data : {temp.shape[0]}\t\n Given value : {self.resolution}")
        
        self.inputsMask = np.zeros((self.__length, 1, self.resolution, self.resolution))
        self.targets = np.zeros((self.__length, 1, self.resolution, self.resolution))
        self.inputsPara = np.zeros((self.__length, 1, 4))

        for i, case in enumerate(self.__traList):
            src = os.path.join(case, os.listdir(case)[0], 'bumpSurfaceData.npz')
            temp = np.load(src)
            self.inputsMask[i, 0] = temp['heights']
            self.targets[i, 0] = temp['pressure']

            # Extract geometry parameters and flow conditions
            temp = case.split('/')
            Mach = float(temp[-1])
            temp = temp[-2].split('_')
            k, c, d = float(temp[0].split('k')[-1]), float(temp[1].split('c')[-1]), float(temp[2].split('d')[-1])
            self.inputsPara[i, 0] = np.array([k, c, d, Mach])

    def _getMean(self):
        
        self.inOffset = 0
        self.tarOffset = 0
        self.inChannels = 1
        self.tarChannels = 1

        for i in range(self.__length):
            self.inOffset += np.sum(self.inputsMask[i,0])
            self.tarOffset += np.sum(self.targets[i,0])

        print(f' Input offset : {self.inOffset}')
        print(f' Target offset : {self.tarOffset}')

        self.inOffset = [self.inOffset / self.__length]
        self.tarOffset = [self.tarOffset / self.__length]

    def _removeOffset(self):
        inOffsetMap = np.ones((self.inChannels, self.resolution, self.resolution))
        for i in range(self.inChannels):
            inOffsetMap[i] *= self.inOffset[i]

        tarOffsetMap = np.ones((self.tarChannels, self.resolution, self.resolution))
        for i in range(self.tarChannels):
            tarOffsetMap[i] *= self.tarOffset[i]

        for i in range(self.__length):
            self.inputsMask[i] -= inOffsetMap
            self.targets[i] -= tarOffsetMap

    def _getNormFactor(self):
        self.inNorm = np.max(np.abs(self.inputsMask[:,0,:,:]))
        self.tarNorm = np.max(np.abs(self.targets[:,0,:,:]))

        print(f' Input scale factor : {self.inNorm}')
        print(f' Target scale factor : {self.tarNorm}')

        self.inNorm = [self.inNorm]
        self.tarNorm = [self.tarNorm]

    def _normalization(self):
        for i in range(self.inChannels):
            self.inputsMask[:,i,:,:] /= self.inNorm[i]
        
        for i in range(self.tarChannels):
            self.targets[:,i,:,:] /= self.tarNorm[i]

if __name__ == '__main__':
    # a = childClass('data/demoData224')
    a = baseDataset('data/demoData256POINT')
    a(resolution=256)
    # b, c, d = a[0]
    # print(b)
    # print(a.mode)
    