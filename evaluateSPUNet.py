import torch, sys, os, glob

sys.path.append('network')

import torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.baseDataset import baseDataset, valBaseDataset, testBaseDataset

####### Training settings ########

# count = 3
count = sys.argv[1]
# Batch size
batchSize = 1
# Network　　
device = torch.device('cpu')
model = f'model/SPUnetEval/{count}'
network = torch.load(model, map_location=device)
# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)

######## Dataset settings ########

# Dataset directory.
dataDir = f'data/trainingData1'
# Dataset usage mode, train or test.
mode = 'train'
caseList='data/trainingData1/caseList1.npz'
# Dataset and the train loader declaration.
dataset = baseDataset(dataDir=dataDir, mode=mode, caseList=caseList)
dataset.preprocessing()
trainLoader = DataLoader(dataset, batchSize, shuffle=True)
testDataset = testBaseDataset(dataDir='data/testingData', trainDataset=dataset)
testDataset.preprocessing()
testLoader = DataLoader(testDataset, batchSize, shuffle=False)

print(testDataset._dataList)

########## Log settings ##########


def evaluate():

    network.eval()
    with torch.no_grad():
        for i, data in enumerate(testLoader):
            inMask, inPara, targets, binaryMask = data
            inMask, inPara, targets = inMask.float().to(device), inPara.float().to(device), targets.float().to(device)
            prediction = network(inMask, inPara)

            inMaskCopy, targetCopy, predictionCopy = inMask.numpy().squeeze(0).copy(), targets.numpy().squeeze(0).copy(), prediction.numpy().squeeze(0).copy()
            np.savez(f'eval{i}_nor', inMasks=inMaskCopy, targets=targetCopy, prediction=predictionCopy)
            binaryMask = binaryMask.squeeze(0)
            dataset.recover(inMaskCopy, targetCopy, predictionCopy, binaryMask)
            np.savez(f'eval{i}', inMasks=inMaskCopy, targets=targetCopy, prediction=predictionCopy)
            if i == 3 : break

def makeDiffImages():
    try:
        resultFolder = sys.argv[2]
    except:
        resultFolder = glob.glob(f'SPUnetEval/net{count}*')[0]
    print(resultFolder)
    for i in range(4):
        path = os.path.join(resultFolder, f'demo{i}')
        if not os.path.exists(path) : os.mkdir(path)
        temp = np.load(f'eval{i}.npz')
        mask, targets, prediction = temp['inMasks'][0], temp['targets'][0], temp['prediction'][0]

        plt.figure()
        plt.contourf(mask, levels=200, cmap='Greys')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'mask'))
        plt.close()

        # M, m = max(np.max(targets), np.max(prediction)), min(np.min(targets), np.min(prediction))
        M, m = np.max(targets), np.min(targets)

        plt.figure()
        plt.contourf(targets, levels=np.linspace(m, M, 200), cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'targets'))
        plt.close()

        plt.figure()
        plt.contourf(prediction, levels=np.linspace(m, M, 200), cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'prediction'))
        plt.close()

        temp = np.abs(np.subtract(targets, prediction))
        # temp /= 100000
        temp = np.divide(temp, targets)
        print(np.max(temp))

        plt.figure()
        plt.contourf(temp, levels=200, cmap='Greens')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'diff'))
        plt.close()

        plt.figure()
        plt.contourf(temp, levels=np.linspace(0, 1, 200), cmap='Greens')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'diff1'))
        plt.close()

        plt.figure()
        plt.contourf(temp, levels=np.linspace(0, 0.5, 200), cmap='Greens')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'diff5'))
        plt.close()

        os.rename(f'eval{i}.npz', os.path.join(path, f'eval{i}.npz'))
        os.rename(f'eval{i}_nor.npz', os.path.join(path, f'eval{i}_nor.npz'))
              
if __name__ == '__main__':
    evaluate()
    makeDiffImages()
    print(network)