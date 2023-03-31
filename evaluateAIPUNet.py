"""
AIP data contains 5 slices : [AIPm2, AIPm1, AIP, AIP1, AIP2].
Each slice contains 6 flow properties : [p, p0, rho, Ux, Uy, Uz].

"""

import torch, sys, os, glob

sys.path.append('network')
sys.path.append('dataset')

import torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.AIPDataset import AIPDataset, testAIPDataset

####### Training settings ########

# count = 3
count = sys.argv[1]
# Batch size
batchSize = 1
# Network　　
device = torch.device('cpu')
model = f'model/AIPUNetEval/{count}'

# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)

######## Dataset settings ########

# Dataset directory.
dataDir = f'data/trainingData1'
# Dataset usage mode, train or test.
caseList='data/trainingData1/caseList1.npz'

########## Log settings ##########


def evaluate():

    network = torch.load(model, map_location=device)

    # Dataset and the train loader declaration.
    dataset = AIPDataset(dataDir=dataDir, mode='train', caseList=caseList)
    dataset.preprocessing()
    trainLoader = DataLoader(dataset, batchSize, shuffle=True)
    testDataset = testAIPDataset(dataDir='data/testingData', trainDataset=dataset)
    testDataset.preprocessing()
    testLoader = DataLoader(testDataset, batchSize, shuffle=False)
    
    print(testDataset._dataList)

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
            # if i == 3 : break

def makeDiffImages():
    try:
        resultFolder = sys.argv[2]
    except:
        resultFolder = glob.glob(f'AIPUNetEval/net{count}*')[0]
    print(resultFolder)
    flowProperties = ['p', 'p0', 'rho', 'Ux', 'Uy', 'Uz']
    for i in range(4):
        rootPath = os.path.join(resultFolder, f'demo{i}')
        if not os.path.exists(rootPath) : os.mkdir(rootPath)
        temp = np.load(f'eval{i}.npz')
        mask, targets, prediction = temp['inMasks'], temp['targets'], temp['prediction']

        plt.figure()
        plt.contourf(mask[0], levels=200, cmap='Greys')
        plt.colorbar()
        plt.savefig(os.path.join(rootPath, 'mask'))
        plt.close()

        for j, flowPropertie in enumerate(flowProperties):
            M, m = max(np.max(targets[j]), np.max(prediction[j])), min(np.min(targets[j]), np.min(prediction[j]))
            # M, m = np.max(targets), np.min(targets)
            subPath = os.path.join(rootPath, flowPropertie)
            if not os.path.exists(subPath) : os.mkdir(subPath)
            plt.figure()
            plt.contourf(targets[j], levels=np.linspace(m, M, 200), cmap='jet')
            plt.colorbar()
            plt.savefig(os.path.join(subPath, 'targets'))
            plt.close()

            plt.figure()
            plt.contourf(prediction[j], levels=np.linspace(m, M, 200), cmap='jet')
            plt.colorbar()
            plt.savefig(os.path.join(subPath, 'prediction'))
            plt.close()

            temp = np.abs(np.subtract(targets[j], prediction[j]))
            temp /= 100000
            # temp = np.divide(temp, targets)
            print(np.max(temp))

            plt.figure()
            plt.contourf(temp, levels=200, cmap='Greens')
            plt.colorbar()
            plt.savefig(os.path.join(subPath, 'diff'))
            plt.close()

            plt.figure()
            plt.contourf(temp, levels=np.linspace(0, 1, 200), cmap='Greens')
            plt.colorbar()
            plt.savefig(os.path.join(subPath, 'diff1'))
            plt.close()

            plt.figure()
            plt.contourf(temp, levels=np.linspace(0, 0.5, 200), cmap='Greens')
            plt.colorbar()
            plt.savefig(os.path.join(subPath, 'diff5'))
            plt.close()

        os.rename(f'eval{i}.npz', os.path.join(rootPath, f'eval{i}.npz'))
        os.rename(f'eval{i}_nor.npz', os.path.join(rootPath, f'eval{i}_nor.npz'))
              
if __name__ == '__main__':
    evaluate()
    makeDiffImages()
    # print(network)