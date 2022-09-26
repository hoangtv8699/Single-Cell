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

from utils import *

# device = torch.device("cuda:0")

dataset_path = 'data/test/multiome/'
pretrain_path = 'pretrain/'

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}gex.h5ad',
    'input_test_mod2': f'{dataset_path}atac.h5ad',
    'subset_pretrain1': f'{pretrain_path}GEX_subset.pkl',
    'subset_pretrain2': f'{pretrain_path}ATAC_subset.pkl',
    'output_pretrain': 'pretrain/',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}

time_train = '26_09_2022 02_08_15'

# if load args
args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net2.pkl', 'rb'))

# get feature type
test_mod1 = sc.read_h5ad(param['input_test_mod1'])
test_mod2 = sc.read_h5ad(param['input_test_mod2'])

# select feature
if param['use_pretrained']:
    subset1 = pk.load(open(param['subset_pretrain1'], 'rb'))
    subset2 = pk.load(open(param['subset_pretrain2'], 'rb'))

params = {'batch_size': 2,
          'shuffle': False,
          'num_workers': 0}
# val_set = ModalityDataset(test_mod1[:, list(subset1)].X, test_mod2.X, types='2mod')
val_set = ModalityDataset(test_mod2.X, test_mod1.X, types='2mod')
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
        rmse += mean_squared_error(label.detach().cpu().numpy(), out.detach().cpu().numpy()) * val_batch.size(0)
        i += 1

rmse = math.sqrt(rmse / len(val_loader.dataset))
print(rmse)
