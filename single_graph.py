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
from sklearn.svm import SVR 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.nn import GCNConv
import re
import itertools
from collections import defaultdict

# 测试对某一种药物用图表征如何
# Node是expr/mut, edge是根据是否是同种cancer种类来构建
pkl_file = open('/mnt/GDSC1/gdsc1.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

data.columns = ['drug', 'cell_line', 'smiles', 'expr', 'Y']


drug = 'Afatinib'
# Use Afatinib to test
drug_data = data[data.drug == 'Afatinib']

# 去除Drug_cellline相同的（但IC50不同）
drug_data.drop_duplicates(subset='cell_line', keep='first', inplace=True)


# expr only
cell_expr = drug_data[['cell_line', 'expr']]
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


X_train, X_test, y_train, y_test = train_test_split(X_top, drug_data['Y'], test_size=0.2, random_state=1)

scaler = MinMaxScaler()
new_X_train = scaler.fit_transform(X_train)
new_X_test = scaler.transform(X_test)

new_X_train = pd.DataFrame(new_X_train)
new_X_train.index = X_train.index 
new_X_test = pd.DataFrame(new_X_test)
new_X_test.index = X_test.index 

# SVM test
def classic_reg(X_train, y_train, X_test, method):
    if method == 'rf':
        rg = RandomForestRegressor()
    if method == 'svr':
        rg = SVR(kernel='rbf')

    rg.fit(X_train, y_train)
    test_pred = rg.predict(X_test)
    return test_pred

def evaluate(y_test, test_pred):
    r2 = r2_score(y_test, test_pred)
    mse = mean_squared_error(y_test, test_pred)   
    pcc = pearsonr(y_test, test_pred)
    return r2, mse, pcc[0], pcc[1]

test_pred = classic_reg(new_X_train, y_train, new_X_test, method='svr')
r2, mse, pcc, pcc_pvalue = evaluate(y_test, test_pred) 
print(r2, mse, pcc, pcc_pvalue)



#### 将Y label投影到0-1之间
def transform_scale(i):
    # ic50 = 1 / (1 + pow(math.exp(float(i)), -0.1))
    ic50 = 1 / (1 + math.exp(-i))
    return ic50
y_train_trans = [transform_scale(i) for i in y_train]
y_test_trans = [transform_scale(i) for i in y_test]
test_pred_trans = classic_reg(new_X_train, y_train_trans, new_X_test, method='svr')
r2, mse, pcc, pcc_pvalue = evaluate(y_test_trans, test_pred_trans) 
print(r2, mse, pcc, pcc_pvalue)


### If we used a single Graph ----

# Data        

# Cell_line_info_file = '/mnt/DeepCDR/data/CCLE/Cell_lines_annotations_20181226.txt'
# cellline2cancertype ={}
# for line in open(Cell_line_info_file).readlines()[1:]:
#     cellline_id = line.split('\t')[1]
#     TCGA_label = line.strip().split('\t')[-1]
#     #if TCGA_label in TCGA_label_set:
#     cellline2cancertype[cellline_id] = TCGA_label

cell_lines = []
for x in np.unique(data.cell_line):
    strinfo = re.compile('[-|\\[|\\]|\\.|_]')
    x = strinfo.sub('', x)
    x = x.upper()
    cell_lines.append(x)

name2depmapid = {}
cellline2cancertype = {}
with open('/mnt/CCLE/sample_info.csv') as f:
    for line in f:
        depmapid = line.split(',')[0]
        if depmapid != 'DepMap_ID':
            name = line.split(',')[2].split('_')[0]
            name2depmapid[name] = depmapid
            cancertype = '_'.join(line.split(',')[2].split('_')[1:])
            cellline2cancertype[name] = cancertype

# Cell-line x Type match
# Only 25 can't match
# c = 0
# for i in cell_lines:
#     if i not in name2depmapid:
#         c += 1
#         print(i)

def build_edge(data, label):
    label = np.array(label)
    cell_lines = data.index
    strinfo = re.compile('[-|\\[|\\]|\\.|_]')
    x = [strinfo.sub('', x).upper() for x in cell_lines]
    types = defaultdict(list)
    edge_index = []
    for i in range(len(x)):
        if x[i] in cellline2cancertype:
            types[cellline2cancertype[x[i]]].append(i)

    for i in types:
        permu = [list(x) for x in itertools.permutations(types[i], 2)]
        edge_index += permu
    
    new_label = []
    for i in label:
        # ic50 = 1 / (1 + pow(math.exp(float(i)), -0.1))
        ic50 = 1 / (1 + math.exp(-i))
        new_label.append(ic50)

    GCNData = DATA.Data(x=torch.Tensor(np.array(data)),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    y=torch.FloatTensor(new_label))
    return GCNData

train_GCNData = build_edge(new_X_train, y_train)
test_GCNData = build_edge(new_X_test, y_test)

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(3000,  256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, 64)
        self.linear = nn.Linear(64,1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        out = self.linear(x)
        out = nn.Sigmoid()(out)
        return out, x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN().to(device)
train_GCNData = train_GCNData.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4)

loss_fn = nn.MSELoss()
model.train()
avg_loss = []
for epoch in range(200):
    optimizer.zero_grad()
    out, _ = model(train_GCNData)
    loss = loss_fn(out, train_GCNData.y)
    loss.backward()
    optimizer.step()
    avg_loss.append(loss.item())
    if not epoch % 10:
        print('Epoch: {}, Loss: {};'.format(epoch, loss.item()))

# Predicting
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

model.eval()
print('Make prediction for {} ...'.format(drug))
with torch.no_grad():
    test_GCNData = test_GCNData.to(device)
    output, _ = model(test_GCNData)

preds = output.cpu().numpy().flatten()
print('RMSE:{}, MSE:{}, PCC:{}, SCC:{};'.format(rmse(y_test, preds), mse(y_test, preds), pearson(y_test, preds), spearman(y_test, preds)))