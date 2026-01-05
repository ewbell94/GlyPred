#!/home/belle6/miniforge3/envs/gppred/bin/python 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import torch.nn.functional as F
from sklearn.model_selection import KFold
from os.path import exists
from features import GlyDataset
from network import AttnNN, MuyGPLayer
from sys import argv
from random import shuffle
from torch.utils.data import Subset, random_split
from torch_geometric.loader import DataLoader

def dist(A, B):
    a2 = torch.sum(A**2, dim=1, keepdim=True)
    b2 = torch.sum(B**2, dim=1, keepdim=True).t()
    ab = A @ B.t()
    d2 = a2-2*ab+b2
    d2 = torch.clamp(d2, min=0.)
    return torch.sqrt(d2)

def trainModel(trainSet,testSet,n_iter_loss=10,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    #torch.set_printoptions(edgeitems=15, sci_mode=False, linewidth=180, precision=3)
    net = AttnNN()
    net = net.to(device)
    gp = MuyGPLayer()
    gp = gp.to(device)
    paramCount = sum(p.numel() for p in net.parameters())
    #print("Total param count: %d"%paramCount)
    #print(len(trainSet),len(testSet))
    optimizer = optim.AdamW(net.parameters(), weight_decay=1e-4)
    epoch = 0 
    testPerf = []
    epoch_loss = []
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=2,cooldown=4,
                                                     threshold=1e-10)

    trainLoader = DataLoader(trainSet, batch_size=512, shuffle=True, num_workers=32, prefetch_factor=10, pin_memory=True)
    if testSet != None:
        testLoader = DataLoader(testSet, batch_size=512, shuffle=True, num_workers=32, pin_memory=True)
    else:
        testLoader = None
    
    totalpoints = torch.tensor(0.)
    ycounts = torch.zeros(29)
    with torch.no_grad():
        net.eval()
        for idx, data in trainLoader:
            ycounts += torch.sum(data.y, dim=0)
            totalpoints += data.y.size(0)
        probs = torch.vstack((totalpoints/(totalpoints-ycounts), totalpoints/ycounts)).to(device)
        net.train()
    #print(probs)
    while epoch < 25:
    #while optimizer.param_groups[0]["lr"] > 1e-5:
        running_loss = 0.
        for idx, data in trainLoader:
            data = data.to(device)
            optimizer.zero_grad()
            outputs = net(data.x,data.edge_index, data.edge_attr, data.batch, data.hotslice)
            errors = F.binary_cross_entropy_with_logits(outputs, data.y, reduction="none")
            #loss = errors.sum()
            loss = (probs.gather(0, data.y.to(torch.long)) * errors).sum()
            #loss = loss + torch.norm(net.nnout.weight, p=2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(ycounts)
        epoch_loss.append(running_loss)
        scheduler.step(running_loss)
        trainX = torch.empty((0,net.embedDim), device=device)
        trainy = torch.empty((0,29), device=device)
        ykey = torch.tensor([-6.,6.], device=device)
        net.eval()
        with torch.no_grad():
            for idx, data in trainLoader:
                data = data.to(device)
                embed = net(data.x, data.edge_index, data.edge_attr, data.batch, data.hotslice)
                trainX = torch.vstack((trainX, embed))
                trainy = torch.vstack((trainy, ykey[data.y.to(torch.long)]))
            gp.trainX = trainX
            gp.trainy = trainy
        gpopt = optim.Adam(gp.parameters(), lr=1e-2)
        for idx, data in trainLoader:
            data = data.to(device)
            with torch.no_grad():
                embed = net(data.x, data.edge_index, data.edge_attr, data.batch, data.hotslice)
            gpopt.zero_grad()
            outputs, var = gp(embed)
            #print(var.min())
            var = torch.clamp(var, min=1e-10)
            errors = (outputs - ykey[data.y.to(torch.long)]) ** 2. / var.unsqueeze(1)
            loss = errors.sum() + 29.*torch.log(var).sum()
            #loss = (probs.gather(0, data.y.to(torch.long)) * errors).sum() + 29.*torch.log(var).sum()
            loss.backward()
            gpopt.step()
        net.train()

        if testLoader != None:
            validLoss, auprcs = auprc(testLoader,net,gp,device)
        else:
            validLoss = 0.
            auprcs = 0.
        testPerf.append(auprcs)
        epoch += 1
        print(f'Epoch {epoch} Training loss: {epoch_loss[-1] :.3f} Validation Loss: {validLoss :.3f}  Validation AUPRC: {sum(testPerf[-1]) :.5f}')
    print("Finished training")
    return testPerf[-1]

def auprc(dl,net,gp,device):
    preds = torch.tensor([])
    labels = torch.tensor([])
    varList = torch.tensor([])
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    net.eval()
    gp.eval()
    loss = 0.
    with torch.no_grad():
        for idx, data in dl:
            data = data.to(device)
            embed = net(data.x, data.edge_index, data.edge_attr, data.batch, data.hotslice)
            outputs, var = gp(embed)
            ytotal = torch.sum(data.y, dim=0)
            loss += criterion(outputs, data.y).item()
            #outputs = torch.sigmoid(outputs)
            preds = torch.cat((preds,outputs.cpu()))
            labels = torch.cat((labels,data.y.cpu()))
            varList = torch.cat((varList, var.cpu()))
    net.train()
    gp.train()
    #print(torch.sigmoid(preds))
    #print(labels)
    #print(var)
    #print(gp.a)
    #print(gp.l)
    #print(gp.ymean)
    preds = torch.sigmoid(preds).numpy()
    labels = labels.numpy()
    totalpos = np.sum(labels, axis=0)
    auprcs = []
    mccs = []
    for i in range(preds.shape[1]):
        if totalpos[i] == 0.:
            auprcs.append(0.)
            mccs.append(0.)
        else:
            #mccs.append(max([skm.matthews_corrcoef(labels[:,i],preds[:,i] > thresh) for thresh in np.arange(0.,1.01,0.01)]))
            mccs.append(skm.matthews_corrcoef(labels[:,i], preds[:,i] > 0.5))
            auprcs.append(skm.average_precision_score(labels[:,i],preds[:,i]))
            #auprcs.append(skm.roc_auc_score(labels[:,i],preds[:,i]))
    f=open("CPLMcsvs.txt")
    for i,line in enumerate(f):
        print(line.strip(), auprcs[i], mccs[i])
    f.close()
    return (loss, auprcs)

def crossValid(ds, fold=5):
    splits = KFold(n_splits=fold, shuffle=True)
    foldRes = []
    for trainIdx, testIdx in splits.split(np.arange(len(ds))):
        print(len(trainIdx),len(testIdx))
        trainSamp = SubsetRandomSampler(trainIdx)
        testSamp = SubsetRandomSampler(testIdx)
        trainLoader = DataLoader(ds, batch_size=128, pin_memory=True, sampler=trainSamp)
        testLoader = DataLoader(ds, batch_size=128, pin_memory=True, sampler=testSamp)
        foldRes.append(trainModel(trainLoader, testLoader))
    print(foldRes)

if __name__=="__main__":
    if len(argv) > 2:
        f=open(argv[2])
        testsitelist = [line.strip() for line in f if line.strip()[-4] == "1"]
        #testsitelist = [line.strip() for line in f]
        f.close()
        f=open(argv[1])
        sitelist = [line.strip() for line in f if line.strip()[-4] == "1" and line.strip() not in testsitelist]
        #sitelist = [line.strip() for line in f if line.strip() not in testsitelist]
        f.close()
        testsitelist = ["test/"+site for site in testsitelist]
        sitelist = ["data/"+site for site in sitelist]
        trainDs = GlyDataset(sitelist)
        testDs = GlyDataset(testsitelist)
    else:
        f=open(argv[1])
        if "test" in argv[1]:
            sitelist = ["test/"+line.strip() for line in f]
        else:
            sitelist = ["data/"+line.strip() for line in f]
        f.close()
        posSites = [site for site in sitelist if site[-4] == "1"]
        negSites = [site for site in sitelist if site[-4] == "0"]
        shuffle(posSites)
        shuffle(negSites)
        trainPos = int(len(posSites)*0.9)
        trainNeg = int(len(negSites)*0.9)
        trainSites = posSites[:trainPos] #+ negSites[:trainNeg]
        testSites = posSites[trainPos:] #+ negSites[trainNeg:]
        trainDs = GlyDataset(trainSites)
        testDs = GlyDataset(testSites)
        #print(len(trainDs),len(testDs))
    perf = trainModel(trainDs,testDs)

