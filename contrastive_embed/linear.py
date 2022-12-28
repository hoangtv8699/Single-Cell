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

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + f' predict'
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
# atac.obs['batch_num'] = atac.obs['batch'].map(batch_dict)

# xoa bot thong tin k dung cho do mat thoi gian tach du lieu
for key in atac.obs_keys():
    del atac.obs[key]

for key in atac.var_keys():
    del atac.var[key]

del atac.uns
del atac.obsm

atac = atac[:42492]
gex = gex[:42492]
sc.pp.log1p(gex)

train_idx, val_idx = train_test_split(np.arange(42492), test_size=0.1, random_state=args.random_seed)
train = atac[train_idx]
label_train = gex[train_idx]
val = atac[val_idx]
label_val = gex[val_idx]

gene_name = 'ACAP3'
train = train[:, gene_locus[gene_name]]
label_train = label_train[:, gene_name]
val = val[:, gene_locus[gene_name]]
label_val = label_val[:, gene_name]

train_gene = torch.tensor(train.X.toarray())
train_type = torch.tensor(train.obs['cell_type_num'])
val_gene = torch.tensor(val.X.toarray())
val_type = torch.tensor(val.obs['cell_type_num'])
train_gene, train_type, val_gene, val_type = train_gene.cuda(), train_type.cuda(), val_gene.cuda(), val_type.cuda()

net = ContrastiveEmbed(train_gene.shape[1])
net.load_state_dict(torch.load(f'{args.save_path}17_12_2022 17_49_55 atac embed/{gene_name}.pkl'))
net.cuda()

out_train = net(train_gene, train_type)
out_val = net(val_gene, val_type)

net = LinearModel()

logger = open(f'{args.logs_path}{time_train}/{gene_name}.log', 'a')
logger.write('net: ' + str(net) + '\n')
params = {'batch_size': 512,
          'shuffle': True,
          'num_workers': 0}

training_set = ModalityDataset2(out_train.detach().cpu().numpy(), label_train.X.toarray())
val_set = ModalityDataset2(out_val.detach().cpu().numpy(), label_val.X.toarray())
train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

best_state_dict = train_predict(train_loader, val_loader, net, args, logger)
torch.save(best_state_dict, f'{args.save_path}{time_train}/{gene_name}.pkl')
print(gene_name)
