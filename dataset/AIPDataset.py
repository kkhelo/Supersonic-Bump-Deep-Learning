"""
name : dataset.py
usage : derive dataset class for AIP prediction implementation
author : Bo-Yuan You
Date : 2023-03-20

"""

import numpy as np
import os, glob
if __name__ == '__main__': 
    from baseDataset import baseDataset
else:
    from .baseDataset import baseDataset


class AIPDataset(baseDataset):
    """
        Derived class for single AIP data. Use AIPDatasetMultiple if want to process multiple channels of AIP data.
        This base class provide : 
            1. basic preprocessing method (package as preprocessing())
            2. recover true value function (recover())

        Return data in sequence:
            1. inputsMask : bump surface heights matrix
            2. inputsPara : geometry parameters and flow condition in (Mach, AIP location, k, c, d) order
            3. targets : AIP data (6, res, res)
            4. binaryMask : AIP geometry mask (1 for solid region, 0 for computational domain)

        Args:
            dataDir : Directory where the dataset is, ex: 'data/demo1'.
            mode : Taining dataset or testing(evaluation) dataset.
            caseList : npz file path, ignored at first time or reorder. 
            res : resolution, raise error when imcompatible with data.
            ratio : ratio of training data to validation data.
            
    """
    def __init__(self, dataDir: str, mode='TRAIN', caseList=None, res=256, ratio: float = 0.8, expandGradient = False) -> None:
        super().__init__(dataDir, mode, caseList, res, ratio, expandGradient)

    def loadData(self, dataList):
        
        self._length = len(dataList)

        # Check if resolution is correct or not
        src = os.path.join(dataList[0], os.listdir(dataList[0])[0], 'AIPData.npz')
        temp = np.load(src)['AIPData']
        if temp.shape[-1] - self.resolution:
            raise ValueError(f"Resolution doesn't math\t\n CFD data : {temp.shape[0]}\t\n Given value : {self.resolution}")
        
        self.inMap = np.zeros((self._length, 4 if self.expandGradient else 1, self.resolution, self.resolution))
        self.binaryMask = np.zeros((self._length, 1, self.resolution, self.resolution))
        self.targets = np.zeros((self._length, temp.shape[1], self.resolution, self.resolution))
        self.inVec = np.zeros((self._length, 5)) 

        for i in range(len(dataList)):
            case = dataList[i]

            # bump surface
            bumpSurfaceData = np.load(os.path.join(case, os.listdir(case)[0], 'bumpSurfaceData.npz'))
            heights = bumpSurfaceData['heights']
            self.inMap[i, :] = self._calculateGradients(heights) if self.expandGradient else heights

            # AIP data
            AIPData = np.load(os.path.join(case, os.listdir(case)[0], 'AIPData.npz'))
            data, tag, mask = AIPData['AIPData'], AIPData['AIPTags'], AIPData['geoMask']
            tag = np.where(tag=='AIP')[0][0]
            self.binaryMask[i, 0] = mask[tag]
            self.targets[i] = data[tag]

            # Extract flow conditions
            temp = case.split('/')
            Mach = float(temp[-1])

            # Extract AIP x coordinate
            locationAIP = np.where(heights == np.max(heights))[0][0]*(0.5/self.resolution)

            temp = temp[-2].split('_')
            k, c, d = float(temp[0].split('k')[-1]), float(temp[1].split('c')[-1]), float(temp[2].split('d')[-1])
            
            # [Mach, AIP Location(x), k, c, d]
            self.inVec[i] = np.array([Mach, locationAIP, k, c, d])

            print(f'Loading -- {i+1:d}/{len(dataList)} completed')

    def _getMean(self):
        self.inChannels = self.inMap.shape[1]
        self.tarChannels = self.targets.shape[1]
        self.inOffset = np.zeros((self.inChannels))
        self.tarOffset = np.zeros((self.tarChannels))

        for i in range(self._length):
            
            for j in range(self.inChannels):
                self.inOffset[j] += np.sum(self.inMap[i,0])/(self.resolution**2*self._length)
            
            validPointAIP = self.resolution**2 - np.sum(self.binaryMask[i,0])
            for j in range(self.tarChannels):
                self.tarOffset[j] += np.sum(self.targets[i,j])/(validPointAIP*self._length)

        print(f' Input offset : ', end='')
        for i in range(self.inChannels) : print(self.inOffset[i], end=' ')
        print()

        print(f' Target offset : ', end='')
        for i in range(self.tarChannels) : print(self.tarOffset[i], end=' ')
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
        super().__init__(dataDir=trainDataset.dataDir, mode='VAL', caseList=trainDataset.caseList, 
                         res=trainDataset.resolution, expandGradient=trainDataset.expandGradient)
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
    def __init__(self, testDataDir, trainDataset:AIPDataset) -> None:
        super().__init__(dataDir=testDataDir, mode='TEST', caseList=None, 
                         res=trainDataset.resolution, expandGradient=trainDataset.expandGradient)
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

    caseList='data/trainingData/caseList1.npz'
    expandGradient = True
    # caseList=None
    tra = AIPDataset('data/trainingData', caseList=caseList, res=256, expandGradient=expandGradient)
    tra.preprocessing()
    val = valAIPDataset(tra)
    val.preprocessing()

    index = 48
    physics = 2

    inMap, _, targets, binaryMask = val[index]
    print(inMap.shape, targets.shape, binaryMask.shape)
    
    inMapCopy, targetsCopy, predCopy = inMap.copy(), targets.copy(), targets.copy()
    val.recover(inMapCopy, targetsCopy, predCopy, binaryMask)

    print(val.inVec[index])

    import matplotlib.pyplot as plt

    plt.figure()
    plt.pcolormesh(inMap[0].transpose(), cmap='Greys')
    plt.colorbar()
    plt.savefig('heightMap')
    plt.close()

    plt.figure()
    plt.pcolormesh(inMap[1].transpose(), cmap='Greys')
    plt.colorbar()
    plt.savefig('gradX')
    plt.close()

    plt.figure()
    plt.pcolormesh(inMap[2].transpose(), cmap='Greys')
    plt.colorbar()
    plt.savefig('gradY')
    plt.close()

    plt.figure()
    plt.pcolormesh(inMap[3].transpose(), cmap='Greys')
    plt.colorbar()
    plt.savefig('gradMag')
    plt.close()

    #####

    plt.figure()
    plt.pcolormesh(targets[physics].transpose(), cmap='Reds')
    plt.colorbar()
    plt.savefig('nor')
    plt.close()

    plt.figure()
    plt.pcolormesh(targetsCopy[physics].transpose(), cmap='Reds')
    plt.colorbar()
    plt.savefig('denor')
    plt.close()    

    temp = glob.glob(os.path.join(val._dataList[index], '*/AIPData.npz'))
    temp = np.load(temp[0])['AIPData']
    
    plt.figure()
    plt.pcolormesh(temp[0][physics].transpose(), cmap='Reds')
    plt.colorbar()
    plt.savefig('ground')
    plt.close()
