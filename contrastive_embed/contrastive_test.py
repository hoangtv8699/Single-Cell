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
atac_paper = sc.read_h5ad('../data/paper data/atac2gex/train_mod1.h5ad')
# gex_paper = sc.read_h5ad('../data/paper data/atac2gex/train_mod2.h5ad')
# gex = sc.read_h5ad(args.label_path)
gene_locus = pk.load(open('../data/multiome/gene locus new.pkl', 'rb'))

gene_list = os.listdir(f'{args.save_path}17_12_2022 17_49_55 atac embed')

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

atac = atac[42492:43492]
# obs_names = []
# for name in atac_paper.obs_names:
#     if name in atac.obs_names:
#         obs_names.append(atac.obs[name].index)
#         break
#
# print(obs_names)

out_final = []

for gene_name in gene_list:
    gene_name = gene_name.split('.')[0]
    if gene_name == 'args net':
        continue
    atac_tmp = atac[:, gene_locus[gene_name]]

    train_gene = torch.tensor(atac_tmp.X.toarray())
    train_type = torch.tensor(atac_tmp.obs['cell_type_num'])
    train_gene, train_type = train_gene.cuda(), train_type.cuda()

    net = ContrastiveEmbed(train_gene.shape[1])
    net.load_state_dict(torch.load(f'{args.save_path}17_12_2022 17_49_55 atac embed/{gene_name}.pkl'))

    # train model by contrastive
    training_set = ModalityDataset(train_gene)
    net.cuda()
    out = net(train_gene, train_type)

    if len(out_final) == 0:
        out_final = np.expand_dims(out.detach().cpu().numpy(), axis=1)
    else:
        out = np.expand_dims(out.detach().cpu().numpy(), axis=1)
        out_final = np.concatenate((out_final, out), axis=1)
    print(gene_name)

pk.dump(out_final, open(f'../data/processed atac test.pkl', 'wb'))

