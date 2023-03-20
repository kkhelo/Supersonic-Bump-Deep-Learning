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

    Args:
        dataDir : Directory where the dataset is, ex: 'data/demo1'.
        mode : Taining dataset or testing(evaluation) dataset.
        caseList : npz file path, ignored at first time or reorder. 
        res : resolution, raise error when imcompatible with data.
        ratio : ratio of training data to validation data.
        workers : number of processor in processing.
        
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
        return self.__length

    def __getitem__(self, index):
        return self.inputsMask[index], self.inputsPara[index], self.targets[index]

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
            caseList += glob.glob(os.path.join(geo, '*'))

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
        print('*** Start data loading step ***')
        self.loadData(self._dataList)
        print(f'*** Data loading step completed in {(time.time() - last):.2f} seconds ***\n')

        last = time.time()
        print('*** Start offset removal step ***')
        if self.mode == self.modeUsage[0]:
            self._getMean()
        self._removeOffset()
        print(f'*** Offset removal step completed in {(time.time()-last):.2f} seconds ***\n')

        last = time.time()
        print('** Start normallization step ***') 
        if self.mode == self.modeUsage[0]:
            self._getNormFactor()
        self._normalization()
        print(f'*** Normalization step completed in {(time.time()-last):.2f} seconds ***\n')

    def loadData(self, dataList):

        self.__length = len(dataList)

        # Check if resolution is correct or not
        src = os.path.join(dataList[0], os.listdir(dataList[0])[0], 'bumpSurfaceData.npz')
        temp = np.load(src)['heights']
        if temp.shape[0] - self.resolution:
            raise ValueError(f"Resolution doesn't math\t\n CFD data : {temp.shape[0]}\t\n Given value : {self.resolution}")
        
        self.inputsMask = np.zeros((self.__length, 1, self.resolution, self.resolution))
        self.targets = np.zeros((self.__length, 1, self.resolution, self.resolution))
        self.inputsPara = np.zeros((self.__length, 4)) 
        # self.inputsPara = np.zeros((self.__length, 4, 1, 1)) 

        for i in range(len(dataList)):
            case = dataList[i]
            src = os.path.join(case, os.listdir(case)[0], 'bumpSurfaceData.npz')
            
            temp = np.load(src)
            self.inputsMask[i, 0] = temp['heights']
            self.targets[i, 0] = temp['pressure']

            # Extract geometry parameters and flow conditions
            temp = case.split('/')
            Mach = float(temp[-1])
            temp = temp[-2].split('_')
            k, c, d = float(temp[0].split('k')[-1]), float(temp[1].split('c')[-1]), float(temp[2].split('d')[-1])
            self.inputsPara[i] = np.array([k, c, d, Mach])

    def _getMean(self):
        
        self.inOffset = 0
        self.tarOffset = 0
        self.inChannels = 1
        self.tarChannels = 1

        for i in range(self.__length):
            self.inOffset += np.sum(self.inputsMask[i,0])
            self.tarOffset += np.sum(self.targets[i,0])

        self.inOffset /= [(self.__length*self.resolution**2)]
        self.tarOffset /= [(self.__length*self.resolution**2)]

        print(f' Input offset : {self.inOffset[0]}')
        print(f' Target offset : {self.tarOffset[0]}')

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

    def recover(self, inputsMask, targets, pred = None, virtual : bool = False):
        """
        function use to recover true data from normalized data
        * virtual argument is used for method overload, prevent value return before further calculation
        """

        inputsMaskCopy, targetsCopy, pred = inputsMask.copy(), targets.copy(), pred.copy()
        for i in range(self.inChannels):
            inputsMaskCopy[i] *= self.inNorm[i]

        for i in range(self.tarChannels):
            targetsCopy[i] *= self.tarNorm[i]
            pred[i] *= self.tarNorm[i]

        inOffsetMap = np.ones((self.inChannels, self.resolution, self.resolution))
        for i in range(self.inChannels):
            inOffsetMap[i] *= self.inOffset[i]
        inputsMaskCopy += inOffsetMap
        
        tarOffsetMap = np.ones((self.tarChannels, self.resolution, self.resolution))
        for i in range(self.tarChannels):
            tarOffsetMap[i] *= self.tarOffset[i]
        targetsCopy += tarOffsetMap
        pred += tarOffsetMap

        if not virtual : 
            return inputsMaskCopy, targetsCopy, pred


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

    Args:
        trainDataset : train dataset class object 
    """
    def __init__(self, dataDir, trainDataset:baseDataset) -> None:
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
    # tra = baseDataset('data/testData', caseList='data/testData/caseList1.npz', res=224)
    tra = baseDataset('data/demoData256', caseList='data/demoData256/caseList1.npz', res=256)
    tra.preprocessing()
    val = valBaseDataset(tra)
    val.preprocessing()
    np.save('pressure', val.targets)
    