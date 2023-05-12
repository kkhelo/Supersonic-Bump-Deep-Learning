"""
AIP data contains 5 slices : [AIPm2, AIPm1, AIP, AIP1, AIP2].
Each slice contains 6 flow properties : [p, p0, rho, Ux, Uy, Uz].

"""

import torch, sys, os, glob, json
import torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.AIPDataset import AIPDataset, testAIPDataset
from network.DimensionalUnet import DimensionalUnet
from utils.networkInfo import countParameters


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
    resultFolder = glob.glob(f'result/AIP/net{count}')[0]
print(resultFolder)

expandGradient, dataDir = None, None
with open(os.path.join(resultFolder, 'trainingInfo.json'), 'r') as of:
    data = json.loads(of.read())
    expandGradient = data['expandGradient']
    dataDir = os.path.join('data', data['training dataset'])

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
        if isinstance(network, DimensionalUnet):
            print(network.__class__.__name__)
            for i, data in enumerate(testLoader):
                inMap, inVec, targets, binaryMask = data
                inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)
                prediction = network(inMap, inVec, binaryMask).numpy().squeeze(0)
                inMap, target, binaryMask = inMap.numpy().squeeze(0), targets.numpy().squeeze(0), binaryMask.numpy().squeeze(0)
                np.savez(f'{dataList[i]}_nor', inMap=inMap, targets=target, prediction=prediction, inVec=inVec)
                inMap, target, prediction = dataset.recover(inMap, target, prediction, binaryMask)
                np.savez(f'{dataList[i]}', inMap=inMap, targets=target, prediction=prediction, inVec=inVec)
        else:
            print(network.__class__.__name__)
            for i, data in enumerate(testLoader):
                inMap, inVec, targets, binaryMask = data
                inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)
                prediction = network(inMap, inVec).numpy().squeeze(0)
                inMap, target, binaryMask = inMap.numpy().squeeze(0), targets.numpy().squeeze(0), binaryMask.numpy().squeeze(0)
                np.savez(f'{dataList[i]}_nor', inMap=inMap, targets=target, prediction=prediction, inVec=inVec)
                inMap, target, prediction = dataset.recover(inMap, target, prediction, binaryMask)
                np.savez(f'{dataList[i]}', inMap=inMap, targets=target, prediction=prediction, inVec=inVec)


    makeDiffImages(len(testLoader), dataList) 

def makeDiffImages(numberOfDemo : int = 1, dataList : list = []):

    flowProperties = ['p', 'p0', 'rho', 'Ux', 'Uy', 'Uz']
    with open(os.path.join(resultFolder, 'networkInfo.txt'), 'w') as of:
        of.write(repr(network))
        of.write('\n\n' + '*'*60+ '\n\n')
        table, totalNUm = countParameters(network)
        of.write(str(table))
        of.write('\n' + str(totalNUm) + '\n')

    for i in range(numberOfDemo):

        path = os.path.join(resultFolder, f'{dataList[i]}')
        if not os.path.exists(path) : os.mkdir(path)
        temp = np.load(f'{dataList[i]}.npz')
        mask, targets, prediction, mach = temp['inMap'], temp['targets'], temp['prediction'], temp['inVec'][0][0]

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