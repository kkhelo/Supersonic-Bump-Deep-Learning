import torch, time, os, sys

sys.path.append('network')
sys.path.append('dataset')

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.baseDataset import baseDataset, valBaseDataset
from dataset.AIPDataset import AIPDataset, valAIPDataset

import torch, network.activatedFunction as af, network.lossFunction as lf
# from network.UNetBase import SPUNet
from network.UNetBaseFinalBlock import SPUNet

####### Training settings ########

# Dataset directory.
dataDir = f'data/trainingData'
# Dataset case list
caseList = os.path.join(dataDir, 'caseList1.npz') 
# scratch 
scratch = True
# Numbers of training epochs
epochs = 3000
# Batch size
batchSize = 16
# Learning rate
lr = 0.0001
# lr = 0.0001
# lr = 0.00001
# lr = 0.000001

# Channel exponent to control network parameters amount
channelBase = 64
inParaLen = 4
# activation
# activation = af.Swish(0.8)
# activation = nn.SELU()
activation = nn.Tanh()
# flinal block
# finalBlockFilters = None
finalBlockFilters = [2, 4]

# number the model
path = 'log/SummaryWriterLog/SPUnet/1'
count = 1
while os.path.exists(path):
    path = f'log/SummaryWriterLog/SPUnet/{count+1:d}'
    count += 1

####### Torch and network settings ########

# Inputs channels, outputs channels
inChannel, outChannel = 1, 1

if scratch:
    network = SPUNet(inChannel=inChannel, outChannel=outChannel, inParaLen=inParaLen, 
                     finalBlockFilters=finalBlockFilters, channelBase=channelBase, activation=activation)
else:
    network = torch.load(f'model/SPUnet/{count-1}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = network.to(device)

torch.set_num_threads(12)
# Loss function
criterion = nn.L1Loss().to(device)
# Optimizer 
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
# Learning rate scheduler
# scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, cycle_momentum=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

######## Dataset settings ########

dataset = baseDataset(dataDir=dataDir, caseList=caseList)
dataset.preprocessing()
trainLoader = DataLoader(dataset, batchSize, shuffle=True, drop_last=True)
valDataset = valBaseDataset(dataset)
valDataset.preprocessing()
valLoader = DataLoader(valDataset, batchSize, shuffle=False)

# directory naming
dirName = f'evalSPUNet/net{count}_{lr}lr'
if scheduler : dirName += f'_{scheduler.__class__.__name__}'
if not scratch : dirName += f'_extendFromNet{count-1}'
dirName += f'_{epochs}epochs_bs{batchSize}_{activation.__class__.__name__}_' + dataDir.split('/')[-1]
if finalBlockFilters:
    dirName += '_finalBlock'
    for fileter in finalBlockFilters : dirName += str(fileter)

os.makedirs(dirName)

########## Log settings ##########

lossHistoryWriter = SummaryWriter(log_dir=path)

########## Training script ##########

def train():
    shortPeriodTime= startTime = time.time()
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        network.train()
        loss_sum = 0

        for _, data in enumerate(trainLoader):
            inMask, inPara, targets, _ = data
            inMask, inPara, targets = inMask.float().to(device), inPara.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            predictions = network(inMask, inPara)
            
            loss = criterion(predictions, targets)
            loss.backward()
            loss_sum += loss.item()

            optimizer.step()
        lossTrain = loss_sum / len(trainLoader)

        if scheduler : scheduler.step()
        loss_val_sum = 0
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(valLoader):
                inMask, inPara, targets, _ = data
                inMask, inPara, targets = inMask.float().to(device), inPara.float().to(device), targets.float().to(device)
                predictions = network(inMask, inPara)
            
                loss = criterion(predictions, targets)
                loss_val_sum += loss.item()
        
        lossVal = loss_val_sum / len(valLoader)

        logLine = f'Epoch {epoch+1:04d} finished | Time duration : {(time.time()-shortPeriodTime):.2f} seconds\n'
        shortPeriodTime = time.time()
        logLine += f'Traning loss : {lossTrain:.4f} | Validation loss : {lossVal:.4f}'
        print(logLine)
        print('-'*30)

        lossHistoryWriter.add_scalars('Loss', {'Train' : lossTrain, 'Validation' : lossVal}, epoch+1)

    totalTime = (time.time()-startTime)/60
    print(f'Training completed | Total time duration : {totalTime:.2f} minutes')
    torch.save(network, f'model/SPUnet/{count}')
       
if __name__ == '__main__':
    try :
        train()
    except Exception as e: 
        print(e)
        os.system(f'rm -rf {path}')
    # train()