import csv 
import pandas as pd
import numpy as np
from torch_geometric import data as DATA
import torch
from preprocess import *
from torch_geometric.data import InMemoryDataset, DataLoader
import random
import torch.nn as nn
from gcn_withoutcl import GCNNet
from scipy import stats
from math import sqrt
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

DPATH = '/mnt/DeepCDR/data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt'%DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv'%DPATH
Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'%DPATH
drug_smiles_file = '/mnt/GraphDRP/data/drug_smiles.csv'


# load Drug
drug_dict, drug_smile, smile_graph = load_drug_smile()

#load gene expression faetures
gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
scaler = MinMaxScaler()
transformed_gexpr = scaler.fit_transform(gexpr_feature)
transformed_gexpr = pd.DataFrame(transformed_gexpr)
transformed_gexpr.index = gexpr_feature.index
transformed_gexpr.columns = gexpr_feature.columns

reader = csv.reader(open(Drug_info_file,'r'))
rows = [item for item in reader]
drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}
drugid2name = {item[0]:item[1] for item in rows if item[5].isdigit()}

# 有重复的Drug name，但ID不同，怀疑批次不同，去掉重复，只取第一个（待验证）
drugid2name, drugid2pubchemid, pubchemid2name = {}, {}, {}
tmp = []
for item in rows:
    if (item[1] not in tmp) and (item[1] in drug_dict):
        drugid2name[item[0]] = item[1]
        drugid2pubchemid[item[0]] = item[5]
        pubchemid2name[item[5]] = item[1]
        tmp.append(item[1])


experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
#filter experiment data
drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
experiment_data_filtered = experiment_data.loc[drug_match_list]

# 以下是新加入的 仅使用有expr信息的cellline
experiment_data_filtered = experiment_data_filtered.loc[:, gexpr_feature.index]


cellline2cancertype ={}
for line in open(Cell_line_info_file).readlines()[1:]:
    cellline_id = line.split('\t')[1]
    TCGA_label = line.strip().split('\t')[-1]
    #if TCGA_label in TCGA_label_set:
    cellline2cancertype[cellline_id] = TCGA_label

# 生成cellline-drug pairs 
data_idx = []
data_dict = {}
for each_drug in experiment_data_filtered.index:
    for each_cellline in experiment_data_filtered.columns:
        # pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]] 
        drug_name = drugid2name[each_drug.split(':')[-1]]
        # if str(pubchem_id) in drugid2pubchemid.values():
        if drug_name in drug_dict:
            if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                data_idx.append((each_cellline, drug_name,ln_IC50,cellline2cancertype[each_cellline])) 
                data_dict[each_cellline + '_' + drug_name] = ln_IC50

nb_celllines = len(set([item[0] for item in data_idx]))
nb_drugs = len(set([item[1] for item in data_idx]))
print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))


# load drug features
def build_data(pairs):
    data_list = []
    for n in pairs:
        celline = n.split('_')[0]
        drug = n.split('_')[1]

        #labels
        ic50 = data_dict[n]
        label = 1 / (1 + pow(math.exp(float(ic50)), -0.1))

        # expr matrix
        expr = transformed_gexpr.loc[celline]

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
num_epoch = 150
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

model_st = 'GCNNet'
dataset = 'GDSC'
model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
loss_fig_name = 'model_' + model_st + '_' + dataset + '_loss'
pearson_fig_name = 'model_' + model_st + '_' + dataset + '_pearson'
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
        torch.save(model.state_dict(), model_file_name)
        with open(result_file_name,'w') as f:
            f.write(','.join(map(str,ret_test)))
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
    plt.savefig(title+".png")  # should before show method

def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title+".png")  # should before show method

draw_loss(train_losses, val_losses, 'loss_expr')
draw_pearson(val_pearsons, 'pearson_expr')


### Test Data test === 可忽略 === 
model.load_state_dict(torch.load('model_GCNNet_GDSC.model'))
model.eval()
G_test,P_test = predicting(model, device, test_loader)
ret_test = print('Test Data - RMSE:{}; MSE:{}; Pearson:{}; Spearman:{}.'.format(rmse(G_test,P_test),mse(G_test,P_test),pearson(G_test,P_test),spearman(G_test,P_test)))
