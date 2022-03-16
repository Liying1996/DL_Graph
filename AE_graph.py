import os
from numpy.core.numeric import indices
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from scipy.stats import median_abs_deviation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv 
from torch_geometric import data as DATA
from preprocess import *
from torch_geometric.data import InMemoryDataset, DataLoader
import random
import torch.nn as nn
from gcn2 import GCNNet
from scipy import stats
from math import sqrt
import math
import matplotlib.pyplot as plt
import pickle
# from tdc.multi_pred import DrugRes

# Load expression data
# gdsc_ic50 = pd.read_csv('/mnt/GDSC2/GDSC2_IC50_after.csv')
# all_express = pd.read_csv('/mnt/GDSC2/expression_after.csv')
# cosmic_express = pd.read_csv('/mnt/GDSC2/expression_cosmic_after.csv')

# Data
# X = all_express.iloc[:,1:]
# y = gdsc_ic50[gdsc_ic50['Drug_ID']=='Cisplatin']['Y'] # don't need

# Load GDSC1 expr file
pkl_file = open('/mnt/GDSC1/gdsc1.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

# Get expr data only
data.columns = ['drug', 'cell_line', 'smiles', 'expr', 'Y']

d = {x:0 for x in np.unique(data['cell_line'])}
count = 0
indices = []
# while count <= len(d.keys()):
for x in range(data.shape[0]):
    cell = data['cell_line'][x]
    if d[cell] == 0:
        indices.append(x)
        d[cell] += 1
        count += 1
    else:
        if count >= len(d.keys()):
            break

cell_expr = data.loc[indices][['cell_line', 'expr']]
cell_expr = cell_expr.reset_index()

def trans_data(data):
    new_data = []
    for i in data['expr']:
        for j in i:
            new_data.append(j)
    return new_data

X = trans_data(cell_expr)
X = np.array(X).reshape(-1,17737)

df_x = pd.DataFrame(X)
df_x.index = cell_expr['cell_line']


# Select top 3k genes (mean absolute deviation)
mad_X = median_abs_deviation(df_x)
top_index = np.argsort(mad_X)[(-3000):]
X_top = df_x.iloc[:,top_index]

# scaler = MinMaxScaler()
# data = scaler.fit_transform(X_top)
# data_loader = DataLoader(dataset=torch.from_numpy(data).float(), batch_size=50)   
X_train, X_test, y_train, y_test = train_test_split(X_top, [0 for x in range(X_top.shape[0])], test_size=0.2, random_state=1)

scaler = MinMaxScaler()
new_X_train = scaler.fit_transform(X_train)
new_X_test = scaler.transform(X_test)

train_loader = DataLoader(dataset=torch.from_numpy(new_X_train).float(), batch_size=50)   
test_loader = DataLoader(dataset=torch.from_numpy(new_X_test).float(), batch_size=50)

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)

# AE model
class AE(nn.Module):
    def __init__(self, input_size):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(512, 256)
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(num_features=512),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(1024, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder,decoder

def AE_train(input_size, train_loader, test_loader):
    model = AE(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  
    all_train_loss = []
    all_test_loss = []

    for epoch in range(200):
        train_loss = 0
        # model.train()
        for x in train_loader:   
            model.train()   
            x = x.cuda()
            _, x_hat = model(x)
            loss = criterion(x_hat, x)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_loss += loss.item()
        
        with torch.no_grad():
            model.eval()
            test_loss = 0

            for x in test_loader:
                # model.eval()
                x = x.cuda()

                _, x_hat = model(x)
                loss = criterion(x_hat, x)

                test_loss += loss.item()

        all_train_loss.append(train_loss/len(train_loader))
        all_test_loss.append(test_loss/len(test_loader))
        if epoch % 5 == 0:
            print('epoch = {}, train loss = {}, test_loss = {};'.format(epoch,\
                                                train_loss/len(train_loader), test_loss/len(test_loader)))
        
    # plt.plot([x for x in range(100)], all_train_loss, label='Train Loss', linewidth=2)
    # plt.plot([y for y in range(100)], all_test_loss, label='Val Loss', linewidth=2)
    # plt.legend()
    # plt.show()
    return model

model = AE_train(X_train.shape[1], train_loader, test_loader)

# train_encoder, _ = model(torch.from_numpy(np.array(new_X_train)).float().cuda())
# test_encoder, _ = model(torch.from_numpy(np.array(new_X_test)).float().cuda())
# X_train_ae = train_encoder.cpu().detach().numpy()
# X_test_ae = test_encoder.cpu().detach().numpy()

# 得到降维之后的Expression Matrix
X_encoder, _ = model(torch.from_numpy(np.array(X_top)).float().cuda())
X_ae = X_encoder.cpu().detach().numpy()
X_ae = pd.DataFrame(X_ae)
X_ae.index = df_x.index

# 以下加上药物的信息

def save_cell_mut_matrix():
    f = open("/mnt/GraphDRP/data/PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[0]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))
    
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1
    
    return cell_dict, cell_feature

cell_dict, cell_feature = save_cell_mut_matrix()

# Depmap ID -> Cell line Name 
depmapid2name = {}
with open('/mnt/DeepCDR/data/cancer_cell_line.info') as f:
    for line in f:
        depmapid = line.split('\t')[0]
        name = line.split('\t')[1]
        depmapid2name[depmapid] = name


DPATH = '/mnt/DeepCDR/data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt'%DPATH
# Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv'%DPATH
# Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'%DPATH
drug_smiles_file = '/mnt/GraphDRP/data/drug_smiles.csv'


# load Drug
drug_dict, drug_smile, smile_graph = load_drug_smile()

# #load gene expression faetures
# gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
# scaler = MinMaxScaler()
# transformed_gexpr = scaler.fit_transform(gexpr_feature)
# transformed_gexpr = pd.DataFrame(transformed_gexpr)
# transformed_gexpr.index = gexpr_feature.index
# transformed_gexpr.columns = gexpr_feature.columns

reader = csv.reader(open(Drug_info_file,'r'))
rows = [item for item in reader]
drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}
drugid2name = {item[0]:item[1] for item in rows if item[5].isdigit()}
drug_names = [item[1] for item in rows if item[5].isdigit()]

# 有重复的Drug name，但ID不同，怀疑批次不同，去掉重复，只取第一个（待验证）
drugid2name, drugid2pubchemid, pubchemid2name = {}, {}, {}
tmp = []
for item in rows:
    if (item[1] not in tmp) and (item[1] in drug_dict):
        drugid2name[item[0]] = item[1]
        drugid2pubchemid[item[0]] = item[5]
        pubchemid2name[item[5]] = item[1]
        tmp.append(item[1])


# 生成cellline-drug pairs 
data_idx = []
data_dict = {}
for i in range(data.shape[0]):
    drug_name = data['drug'][i]
    each_cellline = data['cell_line'][i]
    ln_IC50 = data['Y'][i]
    if (drug_name in smile_graph.keys()) and ((each_cellline + '_' + drug_name) not in data_dict):
        data_idx.append((each_cellline, drug_name,ln_IC50))
        data_dict[each_cellline + '\t' + drug_name] = ln_IC50


nb_celllines = len(set([item[0] for item in data_idx]))
nb_drugs = len(set([item[1] for item in data_idx]))
print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))


# load drug features
def build_data(pairs):
    data_list = []
    for n in pairs:
        cellline_name = n.split('\t')[0]
        # cellline_name = depmapid2name[cellline]
        drug = n.split('\t')[1]

        #labels
        ic50 = data_dict[n]
        label = 1 / (1 + pow(math.exp(float(ic50)), -0.1))

        # expr matrix
        expr = X_ae.loc[cellline_name]

        # drug graph
        c_size, features, edge_index = smile_graph[drug]
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        GCNData = DATA.Data(x=torch.Tensor(features),
                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            y=torch.FloatTensor([label]))

        GCNData.target = torch.FloatTensor([expr])
        data_list.append(GCNData)
    return data_list

# split to train-val-test samples 0.8:0.1:0.1 mixed-test
size1 = int(len(data_dict) * 0.8)
size2 = int(len(data_dict) * 0.9)
keys = list(data_dict.keys())
random.shuffle(keys)
train_data = build_data(keys[:size1])
val_data = build_data(keys[size1:size2])
test_data = build_data(keys[size2:])

train_batch, val_batch, test_batch = 1024, 1024, 1024
train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
print("CPU/GPU: ", torch.cuda.is_available())


## Training Test
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

## Running 
lr = 0.0001
num_epoch = 100
log_interval = 20
print('Learning rate: ', lr)
print('Epochs: ', num_epoch)

train_losses = []
val_losses = []
val_pearsons = []

modeling = GCNNet
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print(device)
model = modeling().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_mse = 1000
best_pearson = 1
best_epoch = -1

for epoch in range(num_epoch):
    train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval)
    G,P = predicting(model, device, val_loader)
    ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
                
    G_test,P_test = predicting(model, device, test_loader)
    ret_test = [rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)]

    train_losses.append(train_loss)
    val_losses.append(ret[1])
    val_pearsons.append(ret[2])

    if ret[1]<best_mse:
        # torch.save(model.state_dict(), model_file_name)
        # with open(result_file_name,'w') as f:
        #     f.write(','.join(map(str,ret_test)))
        best_epoch = epoch+1
        best_mse = ret[1]
        best_pearson = ret[2]
        print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse, 'GCNNet','GDSC')
    else:
        print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, 'GCNNet', 'GDSC')




# Draw plot
def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(title+"_AE.png")  # should before show method

def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title+"_AE.png")  # should before show method

draw_loss(train_losses, val_losses, 'loss')
draw_pearson(val_pearsons, 'pearson')

