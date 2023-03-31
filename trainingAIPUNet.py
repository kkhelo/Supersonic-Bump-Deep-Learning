import torch, time, os, sys, math

sys.path.append('network')
sys.path.append('dataset')

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.AIPDataset import AIPDataset, valAIPDataset

import torch, network.activatedFunction as af, network.lossFunction as lf
# from network.UNetBase import SPUNet
from network.UNetBaseFinalBlock import SPUNet

####### Training settings ########

# scratch 
scratch = False
# Numbers of training epochs
epochs = 10000
# Batch size
batchSize = 16
# Learning rate
# lr = 0.001
# lr = 0.0001
lr = 0.00001
# lr = 0.000001
# Inputs channels, outputs channels
inChannel, out_channel = 1, 6
# Channel exponent to control network parameters amount
channelBase = 64
inParaLen = 5
# activation
# activation = af.Swish(0.8)
# activation = nn.SELU()
activation = nn.Tanh()
# flinal block
# finalBlockFilters = None
finalBlockFilters = [2, 2]

# number the model
path = 'log/SummaryWriterLog/AIP/1'
count = 1
while os.path.exists(path):
    path = f'log/SummaryWriterLog/AIP/{count+1:d}'
    count += 1

# Network　　
if scratch:
    network = SPUNet(inChannel=inChannel, outChannel=out_channel, inParaLen=inParaLen, 
                     finalBlockFilters=finalBlockFilters, channelBase=channelBase, activation=activation)
else:
    network = torch.load(f'model/AIP/{count-1}')
# network = torch.nn.DataParallel(network)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = network.to(device)

# def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
#     input_size = tensor.shape[-2]
#     std = math.sqrt(1/input_size)
#     with torch.no_grad():
#         return tensor.normal_(-std,std)
    
# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         lecun_normal_(m.weight)
#         nn.init.zeros_(m.bias)

# if scratch and weightInit : network.apply(weights_init)

# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)
# Loss function
criterion = nn.L1Loss().to(device)
# Optimizer 
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
# Learning rate scheduler
# scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, cycle_momentum=False)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.7)

######## Dataset settings ########

# Dataset directory.
dataDir = f'data/trainingData1'
# Dataset usage mode, train or test.
mode = 'train'
caseList='data/trainingData1/caseList1.npz'
# Dataset and the train loader declaration.
dataset = AIPDataset(dataDir=dataDir, mode=mode, caseList=caseList)
dataset.preprocessing()
trainLoader = DataLoader(dataset, batchSize, shuffle=True, drop_last=True)
valDataset = valAIPDataset(dataset)
valDataset.preprocessing()
valLoader = DataLoader(valDataset, batchSize, shuffle=False)


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
        # if not epoch % 100 and scheduler : print(f'Learning rate {scheduler.get_last_lr()[0]}')

    totalTime = (time.time()-startTime)/60
    print(f'Training completed | Total time duration : {totalTime:.2f} minutes')
    torch.save(network, f'model/AIP/{count}')
       
if __name__ == '__main__':
    try :
        train()
    except Exception as e: 
        print(e)
        os.system(f'rm -rf {path}')
    # train()