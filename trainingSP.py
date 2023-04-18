import torch, time, os, torch

import torch.nn as nn, network.activatedFunction as af, network.lossFunction as lf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset.baseDataset import baseDataset, valBaseDataset
from network.UNet import UNet
from network.AE import AE

####### Training settings ########

dataDir = 'data/trainingData'
caseList = os.path.join(dataDir, 'caseList1.npz') 
scratch = True
epochs = 1
batchSize = 16
lr = 0.0001
# lr = 0.0001
# lr = 0.00001
# lr = 0.000001

network = UNet
# network = AE

channelFactors = [1,2,2,4,4,8,8,16]

channelBase = 64

finalBlockDivisors = None
# finalBlockDivisors = [2, 4]

# activation = af.Swish(0.8)
# activation = nn.SELU()
activation = nn.Tanh()

expandGradient = False

# ordering the model
path = 'log/SummaryWriterLog/SP/1'
count = 1
while os.path.exists(path):
    path = f'log/SummaryWriterLog/SP/{count+1:d}'
    count += 1

######## Dataset settings ########

dataset = baseDataset(dataDir=dataDir, caseList=caseList)
dataset.preprocessing()
trainLoader = DataLoader(dataset, batchSize, shuffle=True, drop_last=True)
valDataset = valBaseDataset(dataset)
valDataset.preprocessing()
valLoader = DataLoader(valDataset, batchSize, shuffle=False)

####### Torch and network settings ########

inChannel, outChannel = 1, 1
if scratch:
    if network is AE:
        network = AE(inChannel, outChannel, channelBase, channelFactors, dataset.inVec.shape[1], 
                     activation, resolution=dataset.inMap.shape[2], bias=True)
    elif network is UNet:
        network = UNet(inChannel, outChannel, channelBase, channelFactors, dataset.inVec.shape[1], 
                     finalBlockDivisors, activation, resolution=dataset.inMap.shape[2], bias=True)
else:
    network = torch.load(f'model/AIP/{count-1}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = network.to(device)

torch.set_num_threads(12)
criterion = nn.L1Loss().to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
# scheduler = None
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, cycle_momentum=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)

# directory naming
dirName = f'result/SP/net{count}_{network.__class__.__name__}'
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
    torch.save(network, f'model/SP/{count}')
       
if __name__ == '__main__':
    # try :
    #     train()
    # except Exception as e: 
    #     print(e)
    #     os.system(f'rm -rf {path}')
    train()