import pickle as pk
import os
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error


from utils import *

# device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'

dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/processed/{mod1_name}2{mod2_name}/'

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}atac_test_chr1.h5ad',
    'input_test_mod2': f'{dataset_path}gex_test_chr1.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}

time_train = '09_01_2023-23_56_59-atac2gex'

args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

# get feature type
mod1 = sc.read_h5ad(param['input_test_mod1'])
mod2 = sc.read_h5ad(param['input_test_mod2'])

cajal = sc.read_h5ad('../data/paper data/atac2gex/output.h5ad')
cajal_out = cajal[:, mod2.var_names]

mod1 = torch.Tensor(mod1.X.toarray())
mod1 = mod1.cuda()

net = ModalityNET([mod1.shape[1], 256, mod2.shape[1]])
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/GAT.pkl'))
net.cuda()
net.eval()

out = net(mod1)

rmse = cal_rmse(mod2.X, csr_matrix(out.detach().cpu().numpy()))
rmse2 = cal_rmse(mod2.X, cajal_out.X)
print(rmse, rmse2)