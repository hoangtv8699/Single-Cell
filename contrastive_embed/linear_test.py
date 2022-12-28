import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime
from scipy.sparse import csr_matrix


import numpy as np
import torch
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import *

device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/multiome/'

args = Namespace(
    train_path=f'{dataset_path}atac.h5ad',
    label_path=f'{dataset_path}gex.h5ad',
    save_path='../saved_model/',
    logs_path='../logs/',
    epochs=300,
    lr_contras=1e-4,
    patience=10,
    random_seed=17
)

# get feature type
atac = sc.read_h5ad(args.train_path)
gex = sc.read_h5ad(args.label_path)
gene_locus = pk.load(open('../data/multiome/gene locus new.pkl', 'rb'))

gene_list = []
for gene_name in gene_locus.keys():
    if len(gene_locus[gene_name]) > 0:
        gene_list.append(gene_name)

cell_types_dict = {
    'CD8+ T': 0,
    'CD14+ Mono': 1,
    'NK': 2,
    'CD4+ T activated': 3,
    'Naive CD20+ B': 4,
    'Erythroblast': 5,
    'CD4+ T naive': 6,
    'Transitional B': 7,
    'Proerythroblast': 8,
    'CD16+ Mono': 9,
    'B1 B': 10,
    'Normoblast': 11,
    'Lymph prog': 12,
    'G/M prog': 13,
    'pDC': 14,
    'HSC': 15,
    'CD8+ T naive': 16,
    'MK/E prog': 17,
    'cDC2': 18,
    'ILC': 19,
    'Plasma cell': 20,
    'ID2-hi myeloid prog': 21
}
batch_dict = {
    's1d1': 0,
    's1d2': 1,
    's1d3': 2,
    's2d1': 3,
    's2d4': 4,
    's2d5': 5,
    's3d3': 6,
    's3d6': 7,
    's3d7': 8,
    's3d10': 9,
    's4d1': 10,
    's4d8': 11,
    's4d9': 12,
}

# map cell_type sang number
atac.obs['cell_type_num'] = atac.obs['cell_type'].map(cell_types_dict)
# atac.obs['batch_num'] = atac.obs['batch'].map(batch_dict)

# xoa bot thong tin k dung cho do mat thoi gian tach du lieu
for key in atac.obs_keys():
    if key not in ['cell_type_num', 'batch_num', 'cell_type', 'batch']:
        del atac.obs[key]

for key in atac.var_keys():
    del atac.var[key]

del atac.uns
del atac.obsm

atac = atac[41492:42492]
gex = gex[41492:42492]
sc.pp.log1p(gex)

test = atac
label_test = gex

gene_name = 'ACAP3'
test = test[:, gene_locus[gene_name]]
label_test = label_test[:, gene_name]

test_gene = torch.tensor(test.X.toarray())
test_type = torch.tensor(test.obs['cell_type_num'])
train_gene, train_type = test_gene.cuda(), test_type.cuda()

net = ContrastiveEmbed(train_gene.shape[1])
net.load_state_dict(torch.load(f'{args.save_path}17_12_2022 17_49_55 atac embed/{gene_name}.pkl'))
net.cuda()
out_test = net(train_gene, train_type)

net = LinearModel()
net.load_state_dict(torch.load(f'{args.save_path}21_12_2022 23_47_42 predict/{gene_name}.pkl'))
net.cuda()
out_test = net(out_test)

rmse = cal_rmse(label_test.X, csr_matrix(out_test.detach().cpu().numpy()))
print(rmse)
