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


class sliceDataset(baseDataset):
    """
        Derived class for slice data.
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

    def _getDataList(self) -> None:
        print('Start training dataset procedure : ')

        if self.caseList: 
            temp = np.load(self.caseList, allow_pickle=True)
            if temp['ratio'][0] == self.ratio : 
                try :
                    self._dataList = temp['tra']
                    self._dictKey = temp['dictKey']
                    # self.__trainList = []
                    # for case in temp :
                    #     self.__trainList.append(str(case))
                    # self._dataList = self.__trainList
                    return
                except:
                    raise ValueError(f'Unkown case list {self.caseList}, should be a path to the list file.')
                

        cases = glob.glob(os.path.join(self.dataDir, '*/*/*'))
        caseList = []
        nSlice = np.load(os.path.join(cases[0], 'sliceData.npz'))['sliceData'].shape[0]

        for case in cases:
            for i in range(nSlice):
                caseList.append((case, i))

        # Split data
        length = len(caseList)
        np.random.shuffle(caseList)
        sepPoint = int(length*self.ratio)

        self.__trainList = np.array(caseList[:sepPoint+1], dtype=object)
        self.__valList = np.array(caseList[sepPoint+1:], dtype=object)
        ratio = np.array([self.ratio])

        # Save fileList with non-repeat name to base directory
        count = 1
        temp = os.path.join(self.dataDir, 'caseListSlice1.npz')
        while os.path.exists(temp):
            count += 1
            temp = os.path.join(self.dataDir, f'caseListSlice{count}.npz')
            
        np.savez_compressed(temp, tra=self.__trainList, val=self.__valList, dictKey=np.array(cases), ratio=ratio)
        self._dataList = self.__trainList
        self._dictKey = cases
        self.caseList = temp

    def loadData(self, dataList):
        
        self._length = len(dataList)
        
        # Check if resolution is correct or not
        src = os.path.join(dataList[0][0], 'sliceData.npz')
        temp = np.load(src)['sliceData']
        if temp.shape[-1] - self.resolution:
            raise ValueError(f"Resolution doesn't math\t\n CFD data : {temp.shape[0]}\t\n Given value : {self.resolution}")
        
        self.inMap = np.zeros((self._length, 4 if self.expandGradient else 1, self.resolution, self.resolution))
        self.binaryMask = np.zeros((self._length, 1, self.resolution, self.resolution))
        self.targets = np.zeros((self._length, temp.shape[1], self.resolution, self.resolution))
        self.inVec = np.zeros((self._length, 5)) 

        i = 0
        for case in self._dictKey:
            # number of slices
            slices = sorted(dataList[np.where(dataList[:,0] == case),1][0])
            # speed up validation load data process
            if len(slices) == 0 : continue

            # bump surface
            heights = np.load(os.path.join(case, 'bumpSurfaceData.npz'))['heights']

            # slice data for given cfd case 
            sliceData = np.load(os.path.join(case, 'sliceData.npz'))
            data, xCoor, mask = sliceData['sliceData'], sliceData['xCoor'], sliceData['geoMask']
            
            
            for iSlice in slices:
            
                self.inMap[i, :] = self._calculateGradients(heights) if self.expandGradient else heights

                # slice data
                self.binaryMask[i, 0] = mask[iSlice]
                self.targets[i] = data[iSlice]

                # extract flow conditions
                temp = case.split('/')
                Mach = float(temp[-2])

                # extract geometry parameters
                temp = temp[-3].split('_')
                k, c, d = float(temp[0].split('k')[-1]), float(temp[1].split('c')[-1]), float(temp[2].split('d')[-1])
                
                # [Mach, x corrdinate, k, c, d]
                self.inVec[i] = np.array([Mach, xCoor[iSlice], k, c, d])

                print(f'Loading -- {i+1:d}/{len(dataList)} completed')
                i += 1

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

class valSliceDataset(sliceDataset):
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

    def __init__(self, trainDataset:sliceDataset) -> None:
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

        temp = np.load(self.caseList, allow_pickle=True)
        self._dataList = temp['val']
        self._dictKey = temp['dictKey']

class testSliceDataset(sliceDataset):
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
    def __init__(self, testDataDir, trainDataset:sliceDataset) -> None:
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

        cases = glob.glob(os.path.join(self.dataDir, '*/*/*'))
        caseList = []
        nSlice = np.load(os.path.join(cases[0], 'sliceData.npz'))['sliceData'].shape[0]

        for case in cases:
            for i in range(nSlice):
                caseList.append((case, i))

        self._dataList = np.array(caseList, dtype=object)
        self._dictKey = np.array(cases)
        

if __name__ == '__main__':
    import sys
    sys.path.append('dataset')

    caseList='data/trainingData/caseListSlice1.npz'
    expandGradient = False
    # caseList=None
    tra = sliceDataset('data/trainingData', caseList=caseList, res=256, 
                       expandGradient=expandGradient, ratio=0.9)
    tra.preprocessing()
    # tra._getDataList()
    # tra.loadData(tra._dataList)

    val = valSliceDataset(tra)
    val.preprocessing()

    print(tra.targets.shape)
    print(val.targets.shape)

    np.savez('tra', inMap=tra.inMap, tar=tra.targets, bm=tra.binaryMask)
    np.savez('val', inMap=val.inMap, tar=val.targets, bm=val.binaryMask)
    
    