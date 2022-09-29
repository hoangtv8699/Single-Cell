import pickle
import math
import time
import dgl
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import anndata as ad
import logging
import pickle as pk
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from argparse import Namespace
from torch.utils.data import DataLoader
from datetime import datetime
from os.path import exists
from scipy.sparse import csc_matrix

from utils import *

# device = torch.device("cuda:0")

dataset_path = 'data/test/multiome/'
pretrain_path = 'pretrain/'

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}gex.h5ad',
    'input_test_mod2': f'{dataset_path}atac.h5ad',
    'subset_pretrain1': f'{pretrain_path}GEX reducer.pkl',
    'subset_pretrain2': f'{pretrain_path}ATAC reducer.pkl',
    'output_pretrain': 'pretrain/',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}

time_train = '28_09_2022 07_38_01no act_out'

# if load args
args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net2.pkl', 'rb'))

# get feature type
test_mod1 = sc.read_h5ad(param['input_test_mod1'])
test_mod2 = sc.read_h5ad(param['input_test_mod2'])

# select feature
mod1_reducer = pk.load(open(param['subset_pretrain1'], 'rb'))
mod2_reducer = pk.load(open(param['subset_pretrain2'], 'rb'))

params = {'batch_size': 2000,
          'shuffle': False,
          'num_workers': 0}

# log norm train mod1
sc.pp.log1p(test_mod1)

input = csc_matrix(mod2_reducer.transform(test_mod2.X))
label = csc_matrix(mod1_reducer.transform(test_mod1.X))

val_set = ModalityDataset(input, label, types='2mod')
val_loader = DataLoader(val_set, **params)

# args.act_out = 's'

net = ContrastiveModel(args)
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model ATAC param predict.pkl'))
net.cuda()
net.eval()

rmse = 0
i = 0
with torch.no_grad():
    for val_batch, label in val_loader:
        print(i)
        val_batch, label = val_batch.cuda(), label.cuda()
        out = net(val_batch, residual=True, types='predict')
        out_ori = mod1_reducer.inverse_transform(out.detach().cpu().numpy())
        label_ori = mod1_reducer.inverse_transform(label.detach().cpu().numpy())
        rmse += mean_squared_error(label_ori, out_ori) * val_batch.size(0)
        i += 1

rmse = math.sqrt(rmse / len(val_loader.dataset))
print(rmse)

