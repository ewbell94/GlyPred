#!/dors/meilerlab/data/belle6/miniforge3/envs/glypred/bin/python 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch.distributions import Normal, Independent
from torch_geometric.utils import scatter, softmax
from math import sqrt, log
 
class AttnNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedDim = 128
        self.esmEmbed = nn.Linear(2560, self.embedDim)
        self.nodeEmbed = nn.Linear(self.embedDim+1, self.embedDim)
        self.seqEmbed = nn.Linear(28, self.embedDim)
        self.ss = SeqStructBlock(self.embedDim)
        self.norm = nn.BatchNorm1d(self.embedDim)
        self.nnout = nn.Linear(self.embedDim, 29)
        self.inside = 4
        self.outside = 8
            
    def forward(self, x, edge_index, edge_attr, batch, hotslice):
        batches, counts = torch.unique_consecutive(batch, return_counts=True)
        middleSelect = torch.zeros(x.size(0), device=x.device)
        middleSelect[0] = 1.
        middleSelect[torch.cumsum(counts, dim=0)[:-1]] = 1.
        seq = self.seqEmbed(hotslice)
        #seq = self.middle(seq.view(-1,(21*self.embedDim))
        struct = self.esmEmbed(x[:,1:])
        struct = self.nodeEmbed(torch.hstack((struct,x[:,0].unsqueeze(1))))
        seqy = torch.randn_like(seq)
        seqz = torch.randn_like(seq)
        structy = torch.randn_like(struct)
        structz = torch.randn_like(struct)
        with torch.no_grad():
            for n in range(self.outside-1):
                for j in range(self.inside):
                    seqh = seq + seqy + seqz
                    structh = struct + structy + structz
                    seqz, structz = self.ss(seqh, structh, edge_index, edge_attr, middleSelect)
                seqh = seqy + seqz
                structh = structy + structz
                seqy, structy = self.ss(seqh, structh, edge_index, edge_attr, middleSelect)
                
        for j in range(self.inside):
            seqh = seq + seqy + seqz
            structh = struct + structy + structz
            seqz, structz = self.ss(seqh, structh, edge_index, edge_attr, middleSelect)
        seqh = seqy + seqz
        structh = structy + structz
        seqy, structy = self.ss(seqh, structh, edge_index, edge_attr, middleSelect)
        
        seq = seqy[:,10,:]
        struct = structy[middleSelect.to(torch.bool)]
        #print(torch.norm(struct,p=2,dim=1)/torch.norm(seq,p=2,dim=1))
        x = self.norm(seq + struct)
        
        if not self.training:
            return x
        else:
            pred = self.nnout(x)
            return pred

class SeqStructBlock(nn.Module):
    def __init__(self, embedDim, doStruct=True):
        super().__init__()
        self.seqNorm = nn.LayerNorm(embedDim)
        self.structNorm = nn.LayerNorm(embedDim)
        self.dropout = nn.Dropout(p=0.2)

        self.inModel = nn.Sequential(
            nn.Linear(2*embedDim+3,embedDim),
            nn.Sigmoid()
        )
        self.outModel = nn.Sequential(
            nn.Linear(2*embedDim+3,embedDim),
            nn.Sigmoid()
        )
        self.inRand = torch.randn(embedDim, device=torch.device("cuda"))
        self.outRand = torch.randn(embedDim, device=torch.device("cuda"))
        self.seqAttn = nn.LSTM(embedDim, int(embedDim/2), batch_first=True, bidirectional=True)
        
    def forward(self, seq, struct, edge_index, edge_attr, middleSelect):
        seqm = int(seq.size(1)/2)
        seq = self.seqNorm(seq + self.dropout(self.seqAttn(seq)[0]))
        ms = middleSelect.to(torch.bool)
        #struct[ms] = struct[ms] + seq[:,seqm,:]
        sn = struct
        inGate = self.inModel(torch.hstack((sn[edge_index[0,0::2]],sn[edge_index[1,0::2]],edge_attr[1::2])))
        outGate = self.outModel(torch.hstack((sn[edge_index[0,0::2]],sn[edge_index[1,0::2]],edge_attr[0::2])))
        inMessage = scatter(inGate * sn[edge_index[1,0::2]], edge_index[0,0::2], dim_size=struct.size(0))
        struct = struct + self.dropout(inMessage)
        outMessage = scatter(outGate * sn[edge_index[0,0::2]], edge_index[1,0::2], dim_size=struct.size(0))
        struct = struct + self.dropout(outMessage)
        struct = self.structNorm(struct)
        #seq[:,seqm,:] = seq[:,seqm,:] + struct[ms]
        return seq, struct

class MuyGPLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.trainX = None
        self.trainy = None
        self.ymean = nn.Linear(128,29)
        #self.ymean = nn.Parameter(torch.zeros((1,29)))
        #self.ymean = None
        #self.l = nn.Parameter(torch.ones((1,64)))
        self.l = nn.Parameter(torch.tensor(1.))
        self.a = nn.Parameter(torch.tensor(3.))
        self.nn = 128
        
    #This kernel is the RBF kernel (easier to implement than Matern)
    def kernel(self, A, B):
        A = A
        B = B
        d = torch.cdist(A, B)
        #val = self.a * torch.exp(-torch.pow(d, 2.) / (2. * self.l ** 2))
        #val = self.a * (1 + np.sqrt(3) * d / self.l) * torch.exp(-np.sqrt(3) * d / self.l)
        val = self.a * torch.exp(-d / self.l)
        return val
    
    def forward(self, x):
        ymean = self.ymean(x).unsqueeze(1)
        dists = torch.cdist(x, self.trainX)
        if self.training:
            _, neighbors = torch.topk(dists, self.nn+1, largest=False, dim=1)
            nX = self.trainX[neighbors[:,1:]]
            nY = self.trainy[neighbors[:,1:]]
        else:
            _, neighbors = torch.topk(dists, self.nn, largest=False, dim=1)
            nX = self.trainX[neighbors]
            nY = self.trainy[neighbors]
        nY = nY  - ymean + torch.randn_like(nY)
        auto = self.kernel(nX, nX)
        autoCov = torch.linalg.inv(auto)
        crossCov = self.kernel(x.unsqueeze(1), nX)
        kWeights = crossCov @ autoCov
        y = kWeights @ nY
        yVar = self.a * torch.ones(x.size(0), device=x.device) - \
               (kWeights @ crossCov.transpose(1, 2)).squeeze()
        return (y + ymean).squeeze(), yVar
