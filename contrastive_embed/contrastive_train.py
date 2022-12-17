import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

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
    save_path='../saved_model/',
    logs_path='../logs/',
    epochs=300,
    lr_contras=1e-4,
    patience=10,
    random_seed=17
)

# get feature type
atac = sc.read_h5ad(args.train_path)
gene_locus = pk.load(open('../data/multiome/gene locus new.pkl', 'rb'))

gene_list = []
for gene_name in gene_locus.keys():
    if len(gene_locus[gene_name]) > 0:
        gene_list.append(gene_name)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + f' atac embed'
os.mkdir(f'{args.save_path}{time_train}')
os.mkdir(f'{args.logs_path}{time_train}')

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
atac.obs['batch_num'] = atac.obs['batch'].map(batch_dict)

# xoa bot thong tin k dung cho do mat thoi gian tach du lieu
for key in atac.obs_keys():
    if key not in ['cell_type_num', 'batch_num', 'cell_type', 'batch']:
        del atac.obs[key]

for key in atac.var_keys():
    del atac.var[key]

del atac.uns
del atac.obsm

atac = atac[:42492]

train_idx, val_idx = train_test_split(np.arange(42492), test_size=0.1, random_state=args.random_seed)
train = atac[train_idx]
val = atac[val_idx]

for gene_name in gene_list:
    # check if model is trained
    if os.path.exists(f'{args.save_path}{time_train}/{gene_name}.pkl'):
        continue

    train_gene = train[:, gene_locus[gene_name]]
    val_gene = val[:, gene_locus[gene_name]]
    logger = open(f'{args.logs_path}{time_train}/{gene_name}.log', 'a')

    net = ContrastiveEmbed(train_gene.shape[1])
    logger.write('net: ' + str(net) + '\n')
    params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 0}

    # train model by contrastive
    training_set = ModalityDataset(train_gene)
    val_set = ModalityDataset(val_gene)
    train_loader = DataLoader(training_set, **params)
    val_loader = DataLoader(val_set, **params)

    best_state_dict = train_contrastive(train_loader, val_loader, net, args, logger)
    torch.save(best_state_dict, f'{args.save_path}{time_train}/{gene_name}.pkl')
    print(gene_name)

