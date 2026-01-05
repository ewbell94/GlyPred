#!/home/belle6/miniforge3/envs/glypred/bin/python 

import os
import numpy as np
import torch
import urllib.request
import matplotlib.pyplot as plt
import esm
import json
import concurrent.futures as ccf
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.DSSP import DSSP

from random import shuffle

class GlyDataset(Dataset):
    def __init__(self, sitelist):
        self.sitelist = sitelist
        self.newy = None
        
    def __len__(self):
        return len(self.sitelist)

    def __getitem__(self, idx):
        data = torch.load(self.sitelist[idx], weights_only=False)
        if self.newy != None:
            data.y = self.newy[idx].unsqueeze(0)
        return idx, data
        
def findLys(fasta, maxlen):
    try:
        f=open(fasta)
    except:
        print("ERROR: FASTA file not found!")
        exit(1)

    header = ""
    seq = ""
    allLys = {}
    for line in f:
        if line[0] == ">":
            if header == "":
                if seq == "":
                    header = line[1:].strip()
                else:
                    print("WARNING: first sequence had no header, this sequence will not be included")
            else:
                if seq == "":
                    print("WARNING: header with no sequence detected. Skipping...")
                else:
                    if len(seq) < maxlen:
                        seqLys = []
                        for n, letter in enumerate(seq):
                            if letter == "K":
                                seqLys.append(n)
                        allLys[header] = seqLys
                    seq = ""
                    header = line[1:].strip()
        else:
            seq += line.strip()
    f.close()
    if len(seq) < maxlen:
        seqLys = []
        for n, letter in enumerate(seq):
            if letter == "K":
                seqLys.append(n)
        allLys[header] = seqLys
    return allLys
        
def esmEmbed(fasta, maxlen):
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t36_3B_UR50D")
    model.eval()
    model = model.to(device)
    dataset = esm.FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(4096, extra_toks_per_seq=1)
    data_loader = DataLoader(dataset, collate_fn=alphabet.get_batch_converter(maxlen), batch_sampler=batches)
    result = {}
    with torch.no_grad():
        for labels, strs, toks in data_loader:
            onehot = F.one_hot(toks[:,1:],num_classes=28)
            toks = toks.to(device)
            if len(strs[0]) > maxlen or len(strs[0]) < 1000:
                continue
            out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
            for i, label in enumerate(labels):
                result[label] = (out["representations"][model.num_layers][i,1:len(strs[i])+1].clone().cpu(), onehot[i])
                print(label)
    return result
    
def extractPos(seqName, truths, allLys, esms):
    print(seqName)
    features = []
    names = []
    parser = PDBParser()
    try:
        urllib.request.urlretrieve("https://alphafold.ebi.ac.uk/files/AF-%s-F1-model_v4.pdb"%seqName, filename="%s.pdb"%seqName)
        f = open("%s.pdb"%seqName)
        afpose = parser.get_structure(seqName,f)[0]["A"]
        f.close()
        #dssp = DSSP(afpose, "%s.pdb"%seqName, dssp="mkdssp")
        os.remove("%s.pdb"%seqName)
        with urllib.request.urlopen("https://alphafold.ebi.ac.uk/files/AF-%s-F1-predicted_aligned_error_v4.json"%seqName) as url:
            paej = json.load(url)
            pae = paej[0]["predicted_aligned_error"]
    except Exception as e:
        if os.path.exists("%s.pdb"%seqName):
            os.remove("%s.pdb"%seqName)
        print("WARNING: Structure file not found for %s"%seqName)
        print(e)
        return

    nsm = NeighborSearch([a for a in afpose.get_atoms()])
    sr = ShrakeRupley()
    sr.compute(afpose, level="R")
    residues = [res for res in afpose.get_residues()]
    #dsspinds = [dssp[key][0] for key in list(dssp.keys())]
    #dssphot = F.one_hot(dsspinds, num_classes=3)
    if len(residues) != len(esms[seqName][0]):
        print("WARNING: Sequence and structure lengths do not match for %s."%seqName)
        return
    for pos in allLys[seqName]:
        label = [float(seqName in truePos and pos in truePos[seqName]) for truePos in truths]
        if residues[pos].id[1] != pos+1:
            print("WARNING: check model indexing for %s:%d,%d"%(seqName,resind,pos+1))
            return
        if residues[pos].get_resname() != "LYS":
            print("WARNING: Found %s instead of LYS in %s:%d"%(residues[pos].get_resname(),seqName,pos+1))
            return

        windowsize = 10
        hotslice = torch.zeros((windowsize*2+1, esms[seqName][1].size(1)))
        if pos-windowsize < 0.:
            partialslice = esms[seqName][1][0:pos+windowsize+1]
            hotslice[-partialslice.size(0):] = partialslice
        elif pos+windowsize >= esms[seqName][1].size(0):
            partialslice = esms[seqName][1][pos-windowsize:]
            hotslice[:partialslice.size(0)] = partialslice
        else:
            hotslice = esms[seqName][1][pos-windowsize:pos+windowsize+1]

        ca = None
        for a in residues[pos].get_atoms():
            if a.name == "CA":
                ca = a
                break
        if not ca:
            continue
        pocket = nsm.search(ca.coord, 10., level="R")
        pocket.remove(residues[pos])
        pocket.insert(0,residues[pos])
        reject = False
        for res in pocket:
            if reject:
                break
            if next(res.get_atoms()).bfactor < 70.0:
                print("WARNING: Neighborhood of Pos %d of %s has too low PLDDT: %.1f"%(pos+1, seqName,next(res.get_atoms()).bfactor))
                reject = True
                break
            numA = res.id[1]-1
            for otherRes in pocket:
                numB = otherRes.id[1]-1
                if pae[numA][numB] > 5.:
                    print("WARNING: Neighborhood of Pos %d of %s has too high PAE: %.1f"%(pos+1, seqName, pae[numA][numB]))
                    reject = True
                    break
        if reject:
            pocket = [pocket[0]]
        cacoords = torch.empty(0,3)
        nbbcoords = torch.empty(0,3)
        cbbcoords = torch.empty(0,3)
        for res in pocket:
            for a in res.get_atoms():
                if a.name == "CA":
                    cacoords = torch.vstack((cacoords, torch.tensor(a.coord)))
                elif a.name == "N":
                    nbbcoords = torch.vstack((nbbcoords, torch.tensor(a.coord)))
                elif a.name == "C":
                    cbbcoords = torch.vstack((cbbcoords, torch.tensor(a.coord)))
        if cacoords.size(0) != len(pocket):
            print("WARNING: CA missing from some pocket residues")
            continue
        if cacoords.size(0) != nbbcoords.size(0) or cacoords.size(0) != cbbcoords.size(0) or nbbcoords.size(0) != cbbcoords.size(0):
            print("WARNING: Backbone atom counts don't agree.")
            continue
        distmat = torch.cdist(cacoords, cacoords)
        cavecs = cacoords.unsqueeze(1) - cacoords
        nbbvecs = cacoords - nbbcoords
        cbbvecs = cacoords - cbbcoords
        bbnorm = torch.cross(nbbvecs, cbbvecs, dim=1)
        edgeIndex = []
        edgeFeats = []
        resVecs = []
        for i in range(len(pocket)):
            resInd = pocket[i].id[1]-1
            resVecs.append(torch.cat((torch.tensor([pocket[i].sasa]),
                                      #dssphot[resInd],
                                      esms[seqName][0][resInd]
                                      )).to(torch.float))
            for j in range(i+1, len(pocket)):
                if i != 0:
                    break
                if i == j:
                    continue
                edgeIndex.append([i,j])
                edgeIndex.append([j,i])
                forwardAngle = torch.dot(cavecs[i,j],nbbvecs[i])/\
                               (torch.norm(cavecs[i,j])*torch.norm(nbbvecs[i]))
                revAngle = torch.dot(cavecs[j,i],nbbvecs[j])/\
                           (torch.norm(cavecs[j,i])*torch.norm(nbbvecs[j]))
                forwardNorm = torch.cross(nbbvecs[i],cavecs[i,j])
                revNorm = torch.cross(nbbvecs[j],cavecs[j,i])
                forwardDih = torch.dot(bbnorm[i],forwardNorm)/\
                             (torch.norm(bbnorm[i])*torch.norm(forwardNorm))
                revDih = torch.dot(bbnorm[j],revNorm)/\
                         (torch.norm(bbnorm[j])*torch.norm(revNorm))
                edgeFeats.append([distmat[i,j].item(),forwardAngle.item(),forwardDih.item()])
                edgeFeats.append([distmat[j,i].item(),revAngle.item(),revDih.item()])

        #print(seqName,pos,label)
        network = Data(x=torch.vstack(resVecs),
                       edge_index=torch.tensor(edgeIndex, dtype=torch.long).t().view(2,-1),
                       edge_attr=torch.tensor(edgeFeats, dtype=torch.float).view(-1,3),
                       y=torch.tensor(label).unsqueeze(0),
                       hotslice=hotslice.unsqueeze(0).to(torch.float))
        features.append(network)
        names.append("%s_%d_%d"%(seqName,pos,int(1. in label)))
    for i,feat in enumerate(features):
        print(names[i],feat.y)
        torch.save(feat, "data/%s/%s.pt"%(seqName,names[i]))
        
def createFeatureVecs(allLys, labelfiles, esms):
    truths = []
    for labelfile in labelfiles:
        truePos = {}
        try:
            f = open(labelfile)
            for line in f:
                parts = line.strip().split(",")
                if parts[0] not in truePos:
                    truePos[parts[0]] = [int(parts[1])-1]
                else:
                    truePos[parts[0]].append(int(parts[1])-1)
            f.close()
            truths.append(truePos)
        except Exception as e:
            if labelfile != "":
                print("ERROR: Labelfile %s not found!"%labelfile)
                print(e)
                exit(1)
            else:
                print("No labelfile specified, make sure you're testing")
    for seqName in allLys.keys():
        if seqName not in esms.keys():
            continue
        if not os.path.isdir("data/%s"%seqName):
            os.mkdir("data/%s"%seqName)
            extractPos(seqName, truths, allLys, esms)
        esms[seqName] = None
    '''
    with ccf.ThreadPoolExecutor(max_workers=4) as exe:
        futures = []
        for seqName in allLys.keys():
            if not os.path.isdir("data/%s"%seqName) or len(os.listdir("data/%s"%seqName)) == 0:
                if not os.path.isdir("data/%s"%seqName):
                    os.mkdir("data/%s"%seqName)
                futures.append(exe.submit(extractPos, seqName, truths, allLys, esms))

        for future in ccf.as_completed(futures):
            del future
    '''

if __name__ == "__main__":
    from sys import argv
    fasta = argv[1]
    labelfile = argv[2:]
    maxlen = 100000
    allLys = findLys(fasta, maxlen)
    esms = esmEmbed(fasta, maxlen)
    createFeatureVecs(allLys, labelfile, esms)
    #print(self.featureVecs,self.labels)
    '''
        self.pcaModel = PCA(n_components=0.9)
        self.featureVecs = torch.tensor(self.pcaModel.fit_transform(np.array([x.numpy() for x in self.featureVecs])))
        print("Feature vec dimension is now %d"%self.pcaModel.n_components_)
    '''

