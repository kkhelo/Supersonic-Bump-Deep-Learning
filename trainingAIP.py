import torch, time, os, sys, math

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.AIPDataset import AIPDataset, valAIPDataset
import torch, network.activatedFunction as af, network.lossFunction as lf
from network.UNet import UNet
from network.AE import AE
from network.AeroConverter import AeroConverter, GlobalEncoderCNN

####### Training settings ########

dataDir = 'data/trainingData'
caseList = os.path.join(dataDir, 'caseList1.npz') 
scratch = True
epochs = 3000
batchSize = 16
# lr = 0.001
lr = 0.0001
# lr = 0.00001
# lr = 0.000001

network = UNet
# network = AE

converter = AE
globalEncoder = GlobalEncoderCNN
network = AeroConverter

# channelFactors = [1, 2, 2, 4, 4, 8, 8, 16]
channelFactors = [1,2,4,8,16]
channelFactorsConverter = [1,2,4,8]
# expandGradient = True
expandGradient = False
channelBase = 64
channelBaseConverter = 32
# activation = nn.SELU()
activation = nn.Tanh()
# flinal blocks
finalBlockDivisors = None
# finalBlockDivisors = [2, 4]


# ordering the model
path = 'log/SummaryWriterLog/AIP/1'
count = 1
while os.path.exists(path):
    path = f'log/SummaryWriterLog/AIP/{count+1:d}'
    count += 1

######## Dataset settings ########

# Dataset and the train loader declaration.
dataset = AIPDataset(dataDir=dataDir, caseList=caseList, expandGradient=expandGradient)
dataset.preprocessing()
trainLoader = DataLoader(dataset, batchSize, shuffle=True, drop_last=True)
valDataset = valAIPDataset(dataset)
valDataset.preprocessing()
valLoader = DataLoader(valDataset, batchSize, shuffle=False)

####### Torch and network settings ########

inChannel, outChannel = 4 if expandGradient else 1, 6
outChannelConverter = 1

if scratch:
    if network is AE:
        network = AE(inChannel, outChannel, channelBase, channelFactors, dataset.inVec.shape[1], 
                     activation, resolution=dataset.inMap.shape[2], bias=True)
    elif network is UNet:
        network = UNet(inChannel, outChannel, channelBase, channelFactors, dataset.inVec.shape[1], 
                     finalBlockDivisors, activation, resolution=dataset.inMap.shape[2], bias=True)
    elif network is AeroConverter:
        converter = converter(inChannel, outChannelConverter, channelBaseConverter, channelFactorsConverter, 
                              dataset.inVec.shape[1]-1, activation=nn.ReLU(inplace=True), resolution=dataset.inMap.shape[2], bias=True)
        globalEncoder = globalEncoder(inChannel, channelBase, channelFactors)
        network = AeroConverter(outChannel, channelBase, channelFactors, dataset.inVec.shape[1], 
                     converter, globalEncoder, resolution=dataset.inMap.shape[2], bias=True)
else:
    network = torch.load(f'model/AIP/{count-1}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = network.to(device)

torch.set_num_threads(12)
criterion = nn.L1Loss().to(device)
criterionBinaryMask = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
# scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, cycle_momentum=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

# directory naming
dirName = f'result/AIP/net{count}_{network.__class__.__name__}'
for channelsFactor in channelFactors : dirName += str(channelsFactor)
dirName += f'_{lr}lr'
if scheduler : dirName += f'_{scheduler.__class__.__name__}'
dirName += f'_{epochs}epochs_bs{batchSize}_{activation.__class__.__name__}_' + dataDir.split('/')[-1]
if finalBlockDivisors:
    dirName += '_finalBlock'
    for divisor in finalBlockDivisors : dirName += str(divisor)

if expandGradient : dirName += '_expandGradient'

os.makedirs(dirName)

########## Log settings ##########

lossHistoryWriter = SummaryWriter(log_dir=path)

########## Training script ##########

def trainAEorUnet():
    shortPeriodTime= startTime = time.time()
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        network.train()
        L1Sum = 0

        for _, data in enumerate(trainLoader):
            inMap, inVec, targets, _ = data
            inMap, inVec, targets = inMap.float().to(device), inVec.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            predictions = network(inMap, inVec)
            
            loss = criterion(predictions, targets)
            loss.backward()
            L1Sum += loss.item()

            optimizer.step()
        L1Train = L1Sum / len(trainLoader)

        if scheduler : scheduler.step()
        L1ValSum = 0
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(valLoader):
                inMap, inVec, targets, _ = data
                inMap, inVec, targets = inMap.float().to(device), inVec.float().to(device), targets.float().to(device)
                predictions = network(inMap, inVec)
            
                loss = criterion(predictions, targets)
                L1ValSum += loss.item()
        
        
        L1Val = L1ValSum / len(valLoader)

        logLine = f'Epoch {epoch+1:04d} finished | Time duration : {(time.time()-shortPeriodTime):.2f} seconds\n'
        shortPeriodTime = time.time()
        logLine += f'Traning L1 : {L1Train:.4f} | Validation L1 : {L1Val:.4f}'
        print(logLine)
        print('-'*30)

        lossHistoryWriter.add_scalars('L1', {'Train' : L1Train, 'Validation' : L1Val}, epoch+1)

    totalTime = (time.time()-startTime)/60
    print(f'Training completed | Total time duration : {totalTime:.2f} minutes')
    torch.save(network, f'model/AIP/{count}')

def trainAeroConverter():
    shortPeriodTime= startTime = time.time()
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        network.train()
        L1Sum = BCESum = 0

        for _, data in enumerate(trainLoader):
            inMap, inVec, targets, binaryMask = data
            inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)

            optimizer.zero_grad()
            mapPredictions, binaryMaskPrediction = network(inMap, inVec)
            
            L1 = criterion(mapPredictions, targets)
            L1Sum += L1.item()

            BCE = criterionBinaryMask(binaryMaskPrediction, binaryMask)
            BCESum += BCE.item()

            lossTotal = L1 + BCE
            lossTotal.backward()

            optimizer.step()

        L1Train = L1Sum / len(trainLoader)
        BCETrain = BCESum / len(trainLoader)
        TotalTrain = L1Train + BCETrain

        if scheduler : scheduler.step()
        L1ValSum = BCEValSum = 0
        network.eval()
        with torch.no_grad():
            for _, data in enumerate(valLoader):
                inMap, inVec, targets, binaryMask = data
                inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)

                mapPredictions, binaryMaskPrediction = network(inMap, inVec)
            
                L1 = criterion(mapPredictions, targets)
                L1ValSum += L1.item()

                BCE = criterionBinaryMask(binaryMaskPrediction, binaryMask)
                BCEValSum += BCE.item()
        
        
        L1Val = L1ValSum / len(valLoader)
        BCEVal = BCEValSum / len(valLoader)
        TotalVal = L1Val + BCEVal

        logLine = f'Epoch {epoch+1:04d} finished | Time duration : {(time.time()-shortPeriodTime):.2f} seconds\n'
        shortPeriodTime = time.time()
        logLine += f'Traning L1 : {L1Train:.4f} | Validation L1 : {L1Val:.4f}\n'
        logLine += f'Traning BCE : {BCETrain:.4f} | Validation BCE : {BCEVal:.4f}\n'
        logLine += f'Traning Total Loss : {TotalTrain:.4f} | Validation Total Loss : {TotalVal:.4f}'
        print(logLine)
        print('-'*30)

        lossHistoryWriter.add_scalars('L1', {'Train' : L1Train, 'Validation' : L1Val}, epoch+1)
        lossHistoryWriter.add_scalars('BCE', {'Train' : BCETrain, 'Validation' : BCEVal}, epoch+1)
        lossHistoryWriter.add_scalars('Total Loss', {'Train' : TotalTrain, 'Validation' : TotalVal}, epoch+1)

    totalTime = (time.time()-startTime)/60
    print(f'Training completed | Total time duration : {totalTime:.2f} minutes')
    torch.save(network, f'model/AIP/{count}')

if __name__ == '__main__':
    try :
        trainAeroConverter() if isinstance(network, AeroConverter) else trainAEorUnet()
    except Exception as e: 
        print(e)
        os.system(f'rm -rf {path}')
    # train()