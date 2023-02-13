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

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}atac_test_chr1_2.h5ad',
    'input_test_mod2': f'{dataset_path}gex_test_chr1_2.h5ad',
    'test_mod1_domain': f'{dataset_path}atac_test_chr1.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}
# 0.217/0.2152
# time_train = '10_01_2023-15_30_18-atac2gex'
# 0.216/0.2152 
# time_train = '11_01_2023-14_56_52-atac2gex'
time_train = '31_01_2023-14_20_19-atac2gex'

args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

gene_locus = pk.load(open('../craw/gene locus 2.pkl', 'rb'))
gene_dict = pk.load(open('../craw/gene infor 2.pkl', 'rb'))


i = 0
list_gene = []
list_atac = []
for key in gene_locus.keys():
    if len(gene_locus[key]) > 30 and int(gene_dict[key]['chromosome_name'][-3:]) == 1:
        list_gene.append(key)
        list_atac += gene_locus[key]
        i += 1
        if i > 0:
            break

print(len(list_gene))
print(len(list_atac))

# get feature type
mod1 = sc.read_h5ad(param['input_test_mod1'])
# mod1_domain = sc.read_h5ad(param['test_mod1_domain'])
mod1_domain = mod1_domain = mod1[:, list_atac]
mod2 = sc.read_h5ad(param['input_test_mod2'])[:,list_gene]

# svd = pk.load(open(f'{pretrain_path}atac 64.pkl', 'rb'))

# train_total = np.sum(mod1.X.toarray(), axis=1)
# train_batches = set(mod1.obs.batch)
# train_batches_dict = {}
# for batch in train_batches:
#     train_batches_dict[batch] = {}
#     train_batches_dict[batch]['mean'] = np.mean(train_total[mod1.obs.batch == batch])
#     train_batches_dict[batch]['std'] = np.std(train_total[mod1.obs.batch == batch])

# mod1.X = mod1.X.toarray()
# for i in range(mod1.X.shape[0]):
#     mod1.X[i] = (mod1.X[i] - train_batches_dict[mod1.obs.batch[i]]['mean'])/train_batches_dict[mod1.obs.batch[i]]['std']

# mod1.X = csr_matrix(mod1.X)
# # mod1.X = svd.transform(mod1.X)
# mod1 = sc.AnnData(
#     X=svd.transform(mod1.X)
# )

cajal = sc.read_h5ad('../data/paper data/atac2gex/output.h5ad')
cajal_out = cajal[:, mod2.var_names]

# mod1 = torch.Tensor(mod1.X.toarray()).int()
# mod1 = mod1.cuda()

test_set = ModalityDataset2(mod1, mod1_domain, mod2)

params = {'batch_size': 16,
          'shuffle': False,
          'num_workers': 0}

test_loader = DataLoader(test_set, **params)

net = AutoEncoder(mod1.shape[1], mod1_domain.shape[1], mod2.shape[1])
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/GAT.pkl'))
net.cuda()
net.eval()

# out = net(mod1)

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