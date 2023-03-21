"""
name : dataset.py
usage : derive dataset class for AIP prediction implementation
author : Bo-Yuan You
Date : 2023-03-20

"""

import numpy as np
import os, glob
from baseDataset import baseDataset

class AIPDataset(baseDataset):
    """
    Derived class for single AIP data. Use AIPDatasetMultiple if want to process multiple channels of AIP data.
    This base class provide : 
        1. basic preprocessing method (package as preprocessing())
        2. recover true value function (recover())

    Return data in sequence:
        1. inputsMask : bump surface heights matrix
        2. inputsPara : geometry parameters and flow condition in (Mach, AIP location) order
        3. targets : AIP data (6, res, res)
        4. binaryMask : AIP geometry mask (1 for solid region, 0 for computational domain)

    Args:
        dataDir : Directory where the dataset is, ex: 'data/demo1'.
        mode : Taining dataset or testing(evaluation) dataset.
        caseList : npz file path, ignored at first time or reorder. 
        res : resolution, raise error when imcompatible with data.
        ratio : ratio of training data to validation data.
        
    """
    def __init__(self, dataDir: str, mode='TRAIN', caseList=None, res=256, ratio: float = 0.8) -> None:
        super().__init__(dataDir, mode, caseList, res, ratio)

    def loadData(self, dataList):
        
        self._length = len(dataList)

        # Check if resolution is correct or not
        src = os.path.join(dataList[0], os.listdir(dataList[0])[0], 'AIPData.npz')
        temp = np.load(src)['AIPData']
        if temp.shape[-1] - self.resolution:
            raise ValueError(f"Resolution doesn't math\t\n CFD data : {temp.shape[0]}\t\n Given value : {self.resolution}")
        
        self.inputsMask = np.zeros((self._length, 1, self.resolution, self.resolution))
        self.binaryMask = np.zeros((self._length, 1, self.resolution, self.resolution))
        self.targets = np.zeros((self._length, temp.shape[1], self.resolution, self.resolution))
        
        # [Mach, AIP Location(x)]
        self.inputsPara = np.zeros((self._length, 2)) 

        for i in range(len(dataList)):
            print(f'Loading -- {i+1:d}/{len(dataList)} completed')

            case = dataList[i]
            bumpSurfaceDataPath = os.path.join(case, os.listdir(case)[0], 'bumpSurfaceData.npz')
            AIPDataPath = os.path.join(case, os.listdir(case)[0], 'AIPData.npz')
            
            # bump surface
            bumpSurfaceData = np.load(bumpSurfaceDataPath)
            heights = bumpSurfaceData['heights']
            self.inputsMask[i, 0] = heights

            # AIP data
            AIPData = np.load(AIPDataPath)
            data, tag, mask = AIPData['AIPData'], AIPData['AIPTags'], AIPData['geoMask']
            tag = np.where(tag=='AIP')[0][0]
            self.binaryMask[i, 0] = mask[tag]
            self.targets[i] = data[tag]

            # Extract flow conditions
            Mach = float(case.split('/')[-1])

            # Extract AIP x coordinate
            locationAIP = np.where(heights == np.max(heights))[0][0]*(0.5/self.resolution)

            self.inputsPara[i] = np.array([Mach, locationAIP])

    def _getMean(self):
        self.inChannels = 1
        self.tarChannels = self.targets.shape[1]
        self.inOffset = [0]
        self.tarOffset = np.zeros((self.tarChannels))

        for i in range(self._length):
            validPointAIP = self.resolution**2 - np.sum(self.binaryMask[i,0])
            self.inOffset[0] += np.sum(self.inputsMask[i,0])/(self.resolution**2*self._length)
            self.tarOffset[0] += np.sum(self.targets[i,0])/(self.resolution**2*self._length)
            for j in range(1, self.tarChannels):
                self.tarOffset[j] += np.sum(self.targets[i,j])/(validPointAIP*self._length)

        print(f' Input offset : {self.inOffset[0]}')
        print(f' Target offset : ', end='')
        for i in range(self.tarChannels) : print(self.tarOffset[i], end=' ')
        print()

    def _getNormFactor(self):
        self.inNorm = np.max(np.abs(self.inputsMask[:,0,:,:]))
        self.tarNorm = np.zeros((self.tarChannels))
        for i in range(self.tarChannels):
            self.tarNorm[i] = np.max(np.abs(self.targets[:,i,:,:]))

        self.inNorm = [self.inNorm]

        print(f' Input scale factor : {self.inNorm[0]}')
        print(f' Target scale factor : ', end='')
        for i in range(self.tarChannels) : print(self.tarNorm[i], end=' ')
        print()


class valAIPDataset(AIPDataset):
    """
    Validation base dataset derived from AIP dataset
    This base class provide : 
        1. basic preprocessing method using factor from train database
        2. call function SOP 

    Return data in sequence:
        1. inputsMask : bump surface heights matrix
        2. inputsPara : geometry parameters and flow condition in (Mach, AIP location) order
        3. targets : AIP data (6, res, res)
        4. binaryMask : AIP geometry mask (1 for solid region, 0 for computational domain)

    Args:
        trainDataset : train dataset class object 
    """

    def __init__(self, trainDataset:AIPDataset) -> None:
        super().__init__(dataDir=trainDataset.dataDir, mode='VAL', caseList=trainDataset.caseList, res=trainDataset.resolution)
        self.inNorm = trainDataset.inNorm
        self.tarNorm = trainDataset.tarNorm
        self.inOffset = trainDataset.inOffset
        self.tarOffset = trainDataset.tarOffset
        self.inChannels = trainDataset.inChannels
        self.tarChannels = trainDataset.tarChannels
    
    def _getDataList(self) -> None:
        print('Start validation dataset procedure : ')

        temp = np.load(self.caseList)['val']
        self.__valList = []
        for case in temp :
            self.__valList.append(str(case))
        self._dataList = self.__valList


class testAIPDataset(AIPDataset):
    """
    Validation base dataset derived from AIP dataset
    This base class provide : 
        1. basic preprocessing method using factor from train database
        2. call function SOP 

    Return data in sequence:
        1. inputsMask : bump surface heights matrix
        2. inputsPara : geometry parameters and flow condition in (Mach, AIP location) order
        3. targets : AIP data (6, res, res)
        4. binaryMask : AIP geometry mask (1 for solid region, 0 for computational domain)

    Args:
        trainDataset : train dataset class object 
        dataDir : test dataset root directory
    """
    def __init__(self, dataDir, trainDataset:AIPDataset) -> None:
        super().__init__(dataDir=dataDir, mode='TEST', caseList=None, res=trainDataset.resolution)
        self.inNorm = trainDataset.inNorm
        self.tarNorm = trainDataset.tarNorm
        self.inOffset = trainDataset.inOffset
        self.tarOffset = trainDataset.tarOffset
        self.inChannels = trainDataset.inChannels
        self.tarChannels = trainDataset.tarChannels

    def _getDataList(self) -> None:
        print('Start testing dataset procedure : ')

        geometry = glob.glob(os.path.join(self.dataDir, '*'))
        caseList = []
        for geo in geometry :
            caseList += glob.glob(os.path.join(geo, '*'))

        self._dataList = caseList
        

if __name__ == '__main__':
    import sys
    sys.path.append('dataset')

    caseList='data/trainingData1/caseList1.npz'
    # caseList=None
    tra = AIPDataset('data/trainingData1', caseList=caseList, res=256)
    tra.preprocessing()
    val = valAIPDataset(tra)
    val.preprocessing()

    index = 10
    inputsMask, _, targets, binaryMask = val[index]
    print(inputsMask.shape, targets.shape, binaryMask.shape)
    

    inputsMaskCopy, targetsCopy = inputsMask.copy(), targets.copy()
    val.recover(inputsMaskCopy, targetsCopy, targetsCopy, binaryMask)

    print(val.tarOffset)
    print(val.tarNorm)
    print(val.tarChannels)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.contourf(targets[0].transpose(), levels=100, cmap='jet')
    plt.colorbar()
    plt.savefig('nor')
    plt.close()

    plt.figure()
    plt.contourf(targetsCopy[0].transpose(), levels=100, cmap='jet')
    plt.colorbar()
    plt.savefig('denor')
    plt.close()    

    temp = glob.glob(os.path.join(val._dataList[index], '*/AIPData.npz'))
    temp = np.load(temp[0])['AIPData']
    
    plt.figure()
    plt.contourf(temp[0][0].transpose(), levels=100, cmap='jet')
    plt.colorbar()
    plt.savefig('ground')
    plt.close()
    