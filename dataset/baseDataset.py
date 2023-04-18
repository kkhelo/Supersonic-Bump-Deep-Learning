"""
name : dataset.py
usage : base dataset class implementation
author : Bo-Yuan You
Date : 2023-02-21

"""

import numpy as np
import glob, os, time
from torch.utils.data import Dataset


class baseDataset(Dataset):
    """
        Base class for bump simulation data, include data processing.
        This base class provide : 
            1. basic preprocessing method (package as preprocessing())
            2. recover true value function (recover())

        Return data in sequence:
            1. inputsMask : bump surface heights matrix
            2. inputsPara : geometry parameters and flow condition in (k, c, d, Mach) order
            3. targets : surface pressure matrix
            4. binaryMask : all zero mask, for data processing and recovery. Appears here only for conviniece

        Args:
            dataDir : Directory where the dataset is, ex: 'data/demo1'.
            mode : Taining dataset or testing(evaluation) dataset.
            caseList : npz file path, ignored at first time or reorder. 
            res : resolution, raise error when imcompatible with data.
            ratio : ratio of training data to validation data.
    """

    modeUsage = ('TRAIN', 'VAL', 'TEST', 'DEMO')

    def __init__(self, dataDir:str, mode='TRAIN', caseList = None, res = 256, ratio : float = 0.8) -> None:
        super().__init__() 
        self.dataDir = dataDir
        self.caseList = caseList
        self.resolution = res
        self.ratio = ratio
        self.mode = mode.upper()

        if self.mode not in self.modeUsage: 
            raise ValueError(F'Invalid usage mode : {self.mode}, Available Options are (TRAIN, VAL, TEST, DEMO)')

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return self.inMap[index], self.inVec[index], self.targets[index], self.binaryMask[index]

    def _getDataList(self) -> None:
        # Use given caseList
        print('Start training dataset procedure : ')

        if self.caseList : 
            try :
                if self.ratio == float(np.load(self.caseList)['ratio'][0]) : 
                    temp = np.load(self.caseList)['tra']
                    self.__trainList = []
                    for case in temp :
                        self.__trainList.append(str(case))
                    self._dataList = self.__trainList
                    return
            except:
                raise ValueError(f'Unkown case list {self.caseList}, should be a path to the list file.')

        geometry = glob.glob(os.path.join(self.dataDir, '*'))
        caseList = []
        for geo in geometry :
            cases = glob.glob(os.path.join(geo, '*'))
            for case in cases:
                if os.listdir(case):
                    caseList.append(case)

        # Split data
        length = len(caseList)
        np.random.shuffle(caseList)
        sepPoint = int(length*self.ratio)

        self.__trainList = caseList[:sepPoint+1]
        self.__valList = caseList[sepPoint+1:]
        ratio = np.array([self.ratio])

        # Save fileList with non-repeat name to base directory
        count = 1
        temp = os.path.join(self.dataDir, 'caseList1.npz')
        while os.path.exists(temp):
            count += 1
            temp = os.path.join(self.dataDir, f'caseList{count}.npz')
            
        np.savez_compressed(temp, tra=self.__trainList, val=self.__valList, ratio=ratio)
        self._dataList = self.__trainList
        self.caseList = temp

    def preprocessing(self):
        
        # get case list
        self._getDataList()

        # load data
        start = last = time.time()
        print('*** Start data loading step ***\n')
        self.loadData(self._dataList)
        print(f'\n*** Data loading step completed in {(time.time() - last):.2f} seconds ***\n')

        last = time.time()
        print('*** Start offset removal step ***\n')
        if self.mode == self.modeUsage[0]:
            self._getMean()
        self._removeOffset()
        print(f'\n*** Offset removal step completed in {(time.time()-last):.2f} seconds ***\n')

        last = time.time()
        print('** Start normallization step ***\n') 
        
        if self.mode == self.modeUsage[0]:
            self._getNormFactor()
        
        self._normalization()
        
        print(f'\n*** Normalization step completed in {(time.time()-last):.2f} seconds ***\n')
        print(f'\n**** Total time elapsed : {(time.time()-start):.2f} seconds ****\n')

    def loadData(self, dataList):

        self._length = len(dataList)

        # Check if resolution is correct or not
        src = os.path.join(dataList[0], os.listdir(dataList[0])[0], 'bumpSurfaceData.npz')
        temp = np.load(src)['heights']
        if temp.shape[0] - self.resolution:
            raise ValueError(f"Resolution doesn't math\t\n CFD data : {temp.shape[0]}\t\n Given value : {self.resolution}")
        
        self.inMap = np.zeros((self._length, 1, self.resolution, self.resolution))
        self.binaryMask = np.zeros((self._length, 1, self.resolution, self.resolution))
        self.targets = np.zeros((self._length, 1, self.resolution, self.resolution))
        self.inVec = np.zeros((self._length, 4)) 
        
        for i in range(len(dataList)):
            print(f'Loading -- {i+1:d}/{len(dataList)} completed')
            case = dataList[i]
            src = os.path.join(case, os.listdir(case)[0], 'bumpSurfaceData.npz')
            
            temp = np.load(src)
            self.inMap[i, 0] = temp['heights']
            self.targets[i, 0] = temp['pressure']

            # Extract geometry parameters and flow conditions
            temp = case.split('/')
            Mach = float(temp[-1])
            self.inVec[i] = np.array([Mach])

            temp = temp[-2].split('_')
            k, c, d = float(temp[0].split('k')[-1]), float(temp[1].split('c')[-1]), float(temp[2].split('d')[-1])
            self.inVec[i] = np.array([k, c, d, Mach])

    def _getMean(self):
        self.inChannels = 1
        self.tarChannels = 1

        self.inOffset = [np.mean(self.inMap)]
        self.tarOffset = [np.mean(self.targets)]

        print(f' Input offset : {self.inOffset[0]}')
        print(f' Target offset : {self.tarOffset[0]}')

    def _removeOffset(self):
        inOffsetMap = np.ones((self.inChannels, self.resolution, self.resolution))
        for i in range(self.inChannels):
            inOffsetMap[i] *= self.inOffset[i]

        tarOffsetMap = np.ones((self.tarChannels, self.resolution, self.resolution))
        for i in range(self.tarChannels):
            tarOffsetMap[i] *= self.tarOffset[i]

        for i in range(self._length):
            self.inMap[i] -= inOffsetMap
            self.targets[i] -= tarOffsetMap

        # Add value back in mask region
        for i in range(self._length):
            for j in range(self.tarChannels):
                tarOffsetMapToAddBack = self.tarOffset[j] * self.binaryMask[i,0]
                self.targets[i, j] += tarOffsetMapToAddBack

    def _getNormFactor(self):
        self.inNorm = [np.max(np.abs(self.inMap))]
        self.tarNorm = [np.max(np.abs(self.targets))]

        print(f' Input scale factor : {self.inNorm[0]}')
        print(f' Target scale factor : {self.tarNorm[0]}')

    def _normalization(self):
        for i in range(self.inChannels):
            self.inMap[:,i,:,:] /= self.inNorm[i]
        
        for i in range(self.tarChannels):
            self.targets[:,i,:,:] /= self.tarNorm[i]

        # Multiply value back in mask region
        for i in range(self._length):
            for j in range(self.tarChannels):
                tarOffsetMapToMultiplyBack = np.ones((self.resolution, self.resolution))
                tarOffsetMapToMultiplyBack[np.where(self.binaryMask[i,0]==1)] *= self.tarNorm[j]
                self.targets[i,j] *= tarOffsetMapToMultiplyBack

    def recover(self, inMapCopy, targetsCopy, predCopy, binaryMask):
        """
        function use to recover true data from normalized data
        size : (channels, H, W)
        """

        for i in range(self.inChannels):
            inMapCopy[i] *= self.inNorm[i]

        for i in range(self.tarChannels):
            tarOffsetMapToMultiplyBack = np.ones((self.resolution, self.resolution))
            tarOffsetMapToMultiplyBack[np.where(binaryMask[0]==0)] = self.tarNorm[i]

            targetsCopy[i] *= tarOffsetMapToMultiplyBack
            predCopy[i] *= tarOffsetMapToMultiplyBack

        inOffsetMap = np.ones((self.inChannels, self.resolution, self.resolution))
        for i in range(self.inChannels):
            inOffsetMap[i] *= self.inOffset[i]
        inMapCopy += inOffsetMap
        
        for i in range(self.tarChannels):
            tarOffsetMap = np.zeros((self.resolution, self.resolution))
            tarOffsetMap[np.where(binaryMask[0]==0)] = self.tarOffset[i]

            targetsCopy[i] += tarOffsetMap
            predCopy[i] += tarOffsetMap

class valBaseDataset(baseDataset):
    """
        Validation base dataset derived from base dataset
        This base class provide : 
            1. basic preprocessing method using factor from train database
            2. call function SOP 

        Return data in sequence:
            1. inputsMask : bump surface heights matrix
            2. inputsPara : geometry parameters and flow condition in (k, c, d, Mach) order
            3. targets : surface pressure matrix
            4. binaryMask : all zero mask, for data processing and recovery. Appears here only for conviniece

        Args:
            trainDataset : train dataset class object 
    """

    def __init__(self, trainDataset:baseDataset) -> None:
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


class testBaseDataset(baseDataset):
    """
        Validation base dataset derived from base dataset
        This base class provide : 
            1. basic preprocessing method using factor from train database
            2. call function SOP 

        Return data in sequence:
            1. inputsMask : bump surface heights matrix
            2. inputsPara : geometry parameters and flow condition in (k, c, d, Mach) order
            3. targets : surface pressure matrix
            4. binaryMask : all zero mask, for data processing and recovery. Appears here only for conviniece

        Args:
            trainDataset : train dataset class object 
            dataDir : test dataset root directory
    """
    def __init__(self, testDataDir, trainDataset:baseDataset) -> None:
        super().__init__(dataDir=testDataDir, mode='TEST', caseList=None, res=trainDataset.resolution)
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
    # tra = baseDataset('data/testData', caseList='data/testData/caseList1.npz', res=224)
    caseList = 'data/trainingData1/caseList1.npz'
    caseList = None

    tra = baseDataset('data/trainingData', caseList=caseList, res=256)
    tra.preprocessing()
    val = valBaseDataset(tra)
    val.preprocessing()

    index = 10
    inMap, _, targets, binaryMask = val[index]
    
    inMapCopy, targetsCopy, pred  = inMap.copy(), targets.copy(), targets.copy()

    val.recover(inMapCopy, targetsCopy, pred, binaryMask)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.contourf(targets[0], levels=200, cmap='jet')
    plt.colorbar()
    plt.savefig('nor')
    plt.close()

    plt.figure()
    plt.contourf(targetsCopy[0], levels=200, cmap='jet')
    plt.colorbar()
    plt.savefig('denor')
    plt.close()    

    temp = glob.glob(os.path.join(val._dataList[index], '*/bumpSurfaceData.npz'))
    temp = np.load(temp[0])['pressure']

    # temp = np.load(os.path.join(val._dataList[index], '322/bumpSurfaceData.npz'))['pressure']
    
    plt.figure()
    plt.contourf(temp, levels=200, cmap='jet')
    plt.colorbar()
    plt.savefig('ground')
    plt.close()