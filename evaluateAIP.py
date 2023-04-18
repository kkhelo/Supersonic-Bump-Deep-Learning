"""
AIP data contains 5 slices : [AIPm2, AIPm1, AIP, AIP1, AIP2].
Each slice contains 6 flow properties : [p, p0, rho, Ux, Uy, Uz].

"""

import torch, sys, os, glob
import torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.AIPDataset import AIPDataset, testAIPDataset


####### Settings ########

count = sys.argv[1]
# Batch size
batchSize = 1
# Network　　
device = torch.device('cpu')
model = f'model/AIP/{count}'
network = torch.load(model, map_location=device)
# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)

try:
    resultFolder = sys.argv[2]
except:
    resultFolder = glob.glob(f'result/AIP/net{count}_*')[0]
print(resultFolder)

expandGradient = 'expandGradient' in resultFolder
# Dataset directory.
for info in resultFolder.split('_')[::-1]:
    if 'trainingData' in info:
        dataDir = os.path.join('data', info)
        break

print(resultFolder, info, dataDir)
caseList = os.path.join(dataDir, 'caseList1.npz')

########## Log settings ##########

def evaluate():

    # Dataset and the train loader declaration.
    dataset = AIPDataset(dataDir=dataDir, caseList=caseList, expandGradient=expandGradient)
    dataset.preprocessing()
    testDataset = testAIPDataset(testDataDir='data/testingData', trainDataset=dataset)
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
            np.savez(f'{dataList[i]}', inMasks=inMaskCopy, targets=targetCopy, prediction=predictionCopy, inPara=inPara)

    makeDiffImages(len(testLoader), dataList) 

def makeDiffImages(numberOfDemo : int = 1, dataList : list = []):

    flowProperties = ['p', 'p0', 'rho', 'Ux', 'Uy', 'Uz']
    for i in range(numberOfDemo):

        path = os.path.join(resultFolder, f'{dataList[i]}')
        if not os.path.exists(path) : os.mkdir(path)
        temp = np.load(f'{dataList[i]}.npz')
        mask, targets, prediction, mach = temp['inMasks'], temp['targets'], temp['prediction'], temp['inPara'][0][0]

        scaleFactors = [1e5, 1e5*(0.5*1.4*mach**2+1), 1.1614, mach*347.189, mach*347.189, mach*347.189]

        plt.figure()
        plt.contourf(mask[0], levels=200, cmap='Greys')
        plt.colorbar()
        plt.savefig(os.path.join(path, 'mask'))
        plt.close()

        with(open(os.path.join(path, 'info'), 'w')) as of:
            of.write(f'*** Case {dataList[i]} : ***\n\n')
            of.write(f'Scale factors = [')
            for factor in scaleFactors:
                of.write(f' {factor} ')
            of.write(']\n')

            for j in range(len(flowProperties)):
                M1, m1 = np.max(targets[j]), np.min(targets[j])
                M2, m2 = np.max(prediction[j]), np.min(prediction[j])

                M, m = max(M1, M2), min(m1, m2)

                subPath = os.path.join(path, flowProperties[j])
                if not os.path.exists(subPath) : os.mkdir(subPath)

                cmap = plt.cm.get_cmap('jet').copy()
                if j < 3 : cmap.set_under('white')

                plt.figure(figsize=(18,8))
                plt.subplot(121, aspect='equal')
                # plt.imshow(np.rot90(targets[j]), cmap=cmap, vmin=0) if j < 3 else plt.imshow(np.rot90(targets[j]), cmap=cmap)
                plt.imshow(np.rot90(targets[j]), cmap=cmap, vmax=M, vmin=0 if j < 3 else m)
                plt.colorbar()
                plt.axis('off')

                plt.subplot(122, aspect='equal')
                # plt.imshow(np.rot90(prediction[j]), cmap=cmap, vmin=0) if j < 3 else plt.imshow(np.rot90(prediction[j]), cmap=cmap)
                plt.imshow(np.rot90(prediction[j]), cmap=cmap, vmax=M, vmin=0 if j < 3 else m)
                plt.colorbar()
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(subPath, 'ground vs prediction (global scale)'))
                plt.close()

                M, m = M1, m1

                plt.figure(figsize=(18,8))
                plt.subplot(121, aspect='equal')
                # plt.imshow(np.rot90(targets[j]), cmap=cmap, vmin=0) if j < 3 else plt.imshow(np.rot90(targets[j]), cmap=cmap)
                plt.imshow(np.rot90(targets[j]), cmap=cmap, vmax=M, vmin=0 if j < 3 else m)
                plt.colorbar()
                plt.axis('off')

                plt.subplot(122, aspect='equal')
                # plt.imshow(np.rot90(prediction[j]), cmap=cmap, vmin=0) if j < 3 else plt.imshow(np.rot90(prediction[j]), cmap=cmap)
                plt.imshow(np.rot90(prediction[j]), cmap=cmap, vmax=M, vmin=0 if j < 3 else m)
                plt.colorbar()
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(subPath, 'ground vs prediction (ground truth scale)'))
                plt.close()

                temp = np.abs(np.subtract(targets[j], prediction[j])).transpose()
                temp /= scaleFactors[j]
                # temp = np.divide(temp, targets)

                of.write(f'* Channel -- {flowProperties[j]} -- : \n')
                of.write(f' Scale factor : {scaleFactors[j]}\n')
                of.write(f' Max diiference percentages : {np.max(temp):.3f}\n')
                of.write(f' Ground max, min : {M1:.3f}, {m1:.3f}\n')
                of.write(f' Prediction max, min : {M2:.3f}, {m2:.3f}\n')
                

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
                plt.savefig(os.path.join(subPath, 'diff05'))
                plt.close()

                plt.figure()
                plt.contourf(temp, levels=np.linspace(0, 0.1, 200), cmap='Greens')
                plt.colorbar()
                plt.savefig(os.path.join(subPath, 'diff01'))
                plt.close()

        os.rename(f'{dataList[i]}.npz', os.path.join(path, 'denor.npz'))
        os.rename(f'{dataList[i]}_nor.npz', os.path.join(path, 'nor.npz'))

        print(f'Done case {dataList[i]} ----- {i+1}/{len(dataList)}')
              
if __name__ == '__main__':
    evaluate()