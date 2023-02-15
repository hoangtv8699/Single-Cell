import pickle as pk
import os
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from utils import *

# device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'

dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'
chr = 'chr1'

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}{chr}/atac_test.h5ad',
    'input_test_mod2': f'{dataset_path}{chr}/gex_test.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}
# 0.217/0.2152
# time_train = '10_01_2023-15_30_18-atac2gex'
# 0.216/0.2152 
# time_train = '11_01_2023-14_56_52-atac2gex'
# importance path , 0.2125/0.2139
# time_train = '12_02_2023-22_25_23-atac2gex'

# simple path
time_train = '14_02_2023-00_22_52-atac2gex-simple'

# get feature type
mod1 = sc.read_h5ad(param['input_test_mod1'])
mod2 = sc.read_h5ad(param['input_test_mod2'])

with open(f'{param["save_model_path"]}{time_train}/transformation.pkl', "rb") as f:
    info = pk.load(f)

X_test = mod1.X.toarray()
X_test = X_test.T
X_test = (X_test - info["means"]) / info["sds"]
X_test = X_test.T
mod1.X = csr_matrix(X_test)

cajal = sc.read_h5ad('../data/paper data/atac2gex/test_mod2_chr1_cajal.h5ad')
cajal_out = cajal[:, mod2.var_names]

params = {'batch_size': 16,
          'shuffle': False,
          'num_workers': 0}
test_set = ModalityDataset(mod1.X.toarray(), cajal_out.X.toarray(), mod2.X.toarray())
test_loader = DataLoader(test_set, **params)

net = torch.load(f'{param["save_model_path"]}{time_train}/model.pkl')
net.cuda()
net.eval()

outs = []
for mod1_batch, mod1_domain_batch, mod2_batch in test_loader:
    mod1_batch, mod1_domain_batch, mod2_batch = mod1_batch.cuda(), mod1_domain_batch.cuda(), mod2_batch.cuda()
    out = net(mod1_batch, mod1_domain_batch)
    if len(outs) == 0:
        outs = out.detach().cpu().numpy()
    else:
        outs = np.concatenate((outs, out.detach().cpu().numpy()), axis=0)


rmse = cal_rmse(mod2.X, csr_matrix(outs))
rmse2 = cal_rmse(mod2.X, cajal_out.X)
print(rmse, rmse2)
