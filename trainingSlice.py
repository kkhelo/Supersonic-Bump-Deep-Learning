import torch, time, os, sys, math

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.sliceDataset import sliceDataset, valSliceDataset
from network.networkBlock import BottleneckLinear, GlobalEncoderCNN
from network.UNet import UNet
from network.AE import AE
from network.DimensionalUnet import DimensionalUnet
from network.lossFunction import PIContinuityLoss
from utils.helper import trainingRecord

####### Training settings ########

dataDir = 'data/trainingData'
# dataDir = 'data/testingData'
caseList = os.path.join(dataDir, 'caseListSlice1.npz') 
scratch = True
epochs = 3000
batchSize = 16
# lr = 0.001
lr = 0.0001
# lr = 0.00001
# lr = 0.000001

# network = UNet
# network = AE
network = DimensionalUnet


criterionPI = PIContinuityLoss
criterionPI = None

alpha = 0.5

mean = True

if network is AE:
    """ AE """
    channelFactors = [1,2,4,4,8,8,16]
    channelBase = 64

elif network is UNet:
    """ U-net """
    channelFactors = [1,2,4,8,16]
    channelBase = 64

    finalBlockDivisors = None
    finalBlockDivisors = [2, 4]   

elif network is DimensionalUnet:
    """ DimensionalUnet """
    globalEncoder = GlobalEncoderCNN
    bottleneck = BottleneckLinear

    channelBase = 16
    globalChannelBase = 32
    
    channelFactors = [1,2,2,4,4,8,16]
    globalChannelFactors = [1,2,4,8,16]
    
# expandGradient = True
expandGradient = False
bias = True
activation = nn.Tanh()

# ordering the model and
path = 'log/SummaryWriterLog/slice/1'
count = 1
while os.path.exists(path):
    path = f'log/SummaryWriterLog/slice/{count+1:d}'
    count += 1

dirName = f'result/slice/net{count}'
os.makedirs(dirName, exist_ok=True)

######## Dataset settings ########

# Dataset and the train loader declaration.
dataset = sliceDataset(dataDir=dataDir, caseList=caseList, expandGradient=expandGradient)
dataset.preprocessing()
trainLoader = DataLoader(dataset, batchSize, shuffle=True, drop_last=True)
valDataset = valSliceDataset(dataset)
valDataset.preprocessing()
valLoader = DataLoader(valDataset, batchSize, shuffle=False)

####### Torch and network settings ########

inChannel, outChannel = 4 if expandGradient else 1, 6
outChannelConverter = 1
resolution = dataset.inMap.shape[2]
inVectorLength = dataset.inVec.shape[1]

if scratch:
    if network is AE:
        network = AE(inChannel, outChannel, channelBase, channelFactors, inVectorLength, 
                     activation, resolution, bias)
    elif network is UNet:
        network = UNet(inChannel, outChannel, channelBase, channelFactors, inVectorLength, 
                     finalBlockDivisors, activation, resolution, bias)
    elif network is DimensionalUnet:
        globalEncoder = globalEncoder(inChannel, globalChannelBase, globalChannelFactors, resolution, bias)
        bottleneck = bottleneck(globalChannelBase*globalChannelFactors[-1]+channelBase, channelBase*channelFactors[-1])
        network = DimensionalUnet(outChannel, channelBase, channelFactors, inVectorLength, 
                                  globalEncoder, bottleneck, activation, resolution, bias)
else:
    network = torch.load(f'model/slice/{count-1}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
network = network.to(device)

torch.set_num_threads(12)
criterion = nn.L1Loss().to(device)

if criterionPI:
    wallValue = -dataset.tarOffset/dataset.tarNorm
    criterionPI = criterionPI(wallValue, mean).to(device)
    

optimizer = torch.optim.Adam(network.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
# scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, cycle_momentum=False)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)


# training hyper parameters recording

params = {
    'directory name' : dirName,
    'training dataset' : dataDir.split('/')[-1],
    'network' : network.__class__.__name__,
    'channels factors' : channelFactors,
    'channels base' :channelBase,
    'global channels factors' : globalChannelFactors if globalChannelFactors else None,
    'global channels factors' : globalChannelBase if globalChannelBase else None,
    'learning rate' : lr,
    'lr scheduler' : scheduler.__class__.__name__ if scheduler else None,
    'epochs' : epochs,
    'batch size' : batchSize,
    'activation' : activation.__class__.__name__,
    'expandGradient' : expandGradient,
    'physical informed loss' : criterionPI.__class__.__name__,
    'alpha' : alpha if criterionPI else 0
}

trainingRecord(params, os.path.join(dirName, 'trainingInfo.json'))

########## Log settings ##########

lossHistoryWriter = SummaryWriter(log_dir=path)
print(network)

########## Training script ##########

def train():
    shortPeriodTime= startTime = time.time()
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        network.train()
        L1Sum = 0

        for _, data in enumerate(trainLoader):
            inMap, inVec, targets, binaryMask = data
            inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)

            optimizer.zero_grad()
            predictions = network(inMap, inVec, binaryMask) if isinstance(network, DimensionalUnet) else network(inMap, inVec)
            
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
                inMap, inVec, targets, binaryMask = data
                inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)
                predictions = network(inMap, inVec, binaryMask) if isinstance(network, DimensionalUnet) else network(inMap, inVec)
            
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
    torch.save(network, f'model/slice/{count}')

def trainPI():
    shortPeriodTime= startTime = time.time()
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        network.train()
        L1Sum = 0
        PILossSum = 0
        totalLossSum = 0

        for _, data in enumerate(trainLoader):
            inMap, inVec, targets, binaryMask = data
            inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)

            optimizer.zero_grad()
            predictions = network(inMap, inVec, binaryMask) if isinstance(network, DimensionalUnet) else network(inMap, inVec)
            
            L1 = criterion(predictions, targets)

            _, predictionCopy, targetsCopy = dataset.recoverTensor(inMap, predictions, targets, binaryMask)
        
            PILoss = criterionPI(predictionCopy, targetsCopy, binaryMask)

            totalLoss = (1-alpha)*L1 + alpha*PILoss
        
            totalLoss.backward()

            L1Sum += L1.item()
            PILossSum += PILoss.item()
            totalLossSum += totalLoss.item()

            optimizer.step()
        L1Train = L1Sum / len(trainLoader)
        PILossTrain = PILossSum / len(trainLoader)
        totalLossTrain = totalLossSum / len(trainLoader)

        if scheduler : scheduler.step()
        L1ValSum = 0
        PILossValSum = 0
        totalLossValSum = 0
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(valLoader):
                inMap, inVec, targets, binaryMask = data
                inMap, inVec, targets, binaryMask = inMap.float().to(device), inVec.float().to(device), targets.float().to(device), binaryMask.float().to(device)
                predictions = network(inMap, inVec, binaryMask) if isinstance(network, DimensionalUnet) else network(inMap, inVec)
            
                L1 = criterion(predictions, targets)

                _, predictionCopy, targetsCopy = dataset.recoverTensor(inMap, predictions, targets, binaryMask)
                
                

                PILoss = criterionPI(predictionCopy, targetsCopy, binaryMask)

                totalLoss = (1-alpha)*L1 + alpha*PILoss

                L1ValSum += L1.item()
                PILossValSum += PILoss.item()
                totalLossValSum += totalLoss.item()
        
        L1Val = L1ValSum / len(valLoader)
        PILossVal = PILossValSum / len(valLoader)
        totalLossVal = totalLossValSum / len(valLoader)

        logLine = f'Epoch {epoch+1:04d} finished | Time duration : {(time.time()-shortPeriodTime):.2f} seconds\n'
        shortPeriodTime = time.time()
        logLine += f'Traning L1 loss    : {L1Train:.4f}         | Validation L1 loss    : {L1Val:.4f}\n'
        logLine += f'Traning PI loss    : {PILossTrain:.4f}     | Validation PI loss    : {PILossVal:.4f}\n'
        logLine += f'Traning total loss : {totalLossTrain:.4f}  | Validation total loss : {totalLossVal:.4f}'
        print(logLine)
        print('-'*45)

        lossHistoryWriter.add_scalars('L1', {'Train' : L1Train, 'Validation' : L1Val}, epoch+1)
        lossHistoryWriter.add_scalars('PI Loss', {'Train' : PILossTrain, 'Validation' : PILossVal}, epoch+1)
        lossHistoryWriter.add_scalars('total loss', {'Train' : totalLossTrain, 'Validation' : totalLossVal}, epoch+1)

    totalTime = (time.time()-startTime)/60
    print(f'Training completed | Total time duration : {totalTime:.2f} minutes')
    torch.save(network, f'model/slice/{count}')


if __name__ == '__main__':
    try :
        trainPI() if criterionPI else train()
    except Exception as e: 
        print(e)
        os.system(f'rm -rf {path}')
    # trainPI()