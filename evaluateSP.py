import torch, sys, os, glob

sys.path.append('network')

import torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.baseDataset import baseDataset, valBaseDataset, testBaseDataset

####### Settings ########

count = sys.argv[1]
# Batch size
batchSize = 1
# Network　　
device = torch.device('cpu')
model = f'model/SP/{count}'
network = torch.load(model, map_location=device)
# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)

try:
    resultFolder = sys.argv[2]
except:
    resultFolder = glob.glob(f'result/SP/net{count}*')[0]
print(resultFolder)

######## Dataset settings ########

# Dataset directory.
for info in resultFolder.split('_')[::-1]:
    if 'trainingData' in info:
        dataDir = os.path.join('data', info)
        break

# Dataset usage mode, train or test.
caseList = os.path.join(dataDir, 'caseList1.npz')

########## Log settings ##########

def evaluate():

    # Dataset and the train loader declaration.
    dataset = baseDataset(dataDir=dataDir, caseList=caseList)
    dataset.preprocessing()
    testDataset = testBaseDataset(testDataDir='data/testingData', trainDataset=dataset)
    testDataset.preprocessing()
    testLoader = DataLoader(testDataset, batchSize, shuffle=False)

    dataList = list()
    for dataPath in testDataset._dataList : dataList.append('_'.join(dataPath.split('/')[-2:]))
    print(testDataset._dataList)

    network.eval()
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            inMask, inPara, targets, binaryMask = data
            inMask, inPara, targets = inMask.float().to(device), inPara.float().to(device), targets.float().to(device)
            prediction = network(inMask, inPara)

            inMaskCopy, targetCopy, predictionCopy = inMask.numpy().squeeze(0).copy(), targets.numpy().squeeze(0).copy(), prediction.numpy().squeeze(0).copy()
            np.savez(f'{dataList[i]}_nor', inMasks=inMaskCopy, targets=targetCopy, prediction=predictionCopy)
            binaryMask = binaryMask.squeeze(0)
            dataset.recover(inMaskCopy, targetCopy, predictionCopy, binaryMask)
            np.savez(f'{dataList[i]}', inMasks=inMaskCopy, targets=targetCopy, prediction=predictionCopy)

    makeDiffImages(len(testLoader), dataList) 

def makeDiffImages(numberOfDemo : int = 1, dataList : list = []):
    
    for i in range(numberOfDemo):
        path = os.path.join(resultFolder, f'{dataList[i]}')
        if not os.path.exists(path) : os.mkdir(path)
        temp = np.load(f'{dataList[i]}.npz')
        mask, targets, prediction = temp['inMasks'][0], temp['targets'][0], temp['prediction'][0]

        plt.figure()
        plt.contourf(mask, levels=200, cmap='Greys')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'mask'))
        plt.close()

        # M, m = max(np.max(targets), np.max(prediction)), min(np.min(targets), np.min(prediction))
        M, m = np.max(targets), np.min(targets)

        plt.figure(figsize=(18,8))
        plt.subplot(121, aspect='equal')
        plt.contourf(targets, levels=np.linspace(m, M, 200), cmap='jet')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(122, aspect='equal')
        plt.contourf(prediction, levels=np.linspace(m, M, 200), cmap='jet')
        plt.colorbar()
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'ground vs prediction'))
        plt.close()

        temp = np.abs(np.subtract(targets, prediction))
        # temp /= 100000
        temp = np.divide(temp, targets)
        print(np.max(temp))

        plt.figure()
        plt.contourf(temp, levels=200, cmap='Greens')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'diff'))
        plt.close()

        plt.figure()
        plt.contourf(temp, levels=np.linspace(0, 1, 200), cmap='Greens')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'diff1'))
        plt.close()

        plt.figure()
        plt.contourf(temp, levels=np.linspace(0, 0.5, 200), cmap='Greens')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(os.path.join(path, 'diff5'))
        plt.close()

        os.rename(f'{dataList[i]}.npz', os.path.join(path, 'denor.npz'))
        os.rename(f'{dataList[i]}_nor.npz', os.path.join(path, 'nor.npz'))
              
if __name__ == '__main__':
    evaluate()