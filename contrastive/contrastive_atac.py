import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

import torch
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import *

device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/multiome/'

param = {
    'use_pretrained': True,
    'input_train_mod1': f'{dataset_path}atac.h5ad',
    'input_train_mod2': f'{dataset_path}gex.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}

args = Namespace(
    num_class=22,
    latent_feats=16,
    pred_hid_feats=256,
    embed_hid_feats=256,
    random_seed=17,
    activation='relu',
    act_out='relu',
    num_embed_layer=2,
    num_pred_layer=2,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=300,
    lr_contras=1e-4,
    lr_ae=1e-4,
    lr_pred=1e-4,
    normalization='batch',
    patience=10
)

# get feature type
train_mod1 = sc.read_h5ad(param['input_train_mod1'])[:]
train_mod2 = sc.read_h5ad(param['input_train_mod2'])[:]

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + f' {mod1_name} to {mod2_name}'
# time_train = '03_10_2022 22_30_28 gex to atac'
os.mkdir(f'{param["save_model_path"]}{time_train}')
logger = open(f'{param["logs_path"]}{time_train}.log', 'a')

# if not using reducer
mod1 = train_mod1.X
mod2 = train_mod2.X

args.input_feats1 = mod1.shape[1]
args.input_feats2 = mod2.shape[1]
args.out_feats1 = mod1.shape[1]
args.out_feats2 = mod2.shape[1]

logger.write('args1: ' + str(args) + '\n')
# dump args
pk.dump(args, open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'wb'))

mod1_train, mod1_val, mod2_train, mod2_val = train_test_split(mod1, mod2, test_size=0.1, random_state=args.random_seed)

mod1_train = sc.AnnData(mod1_train, dtype=mod1_train.dtype)
mod1_val = sc.AnnData(mod1_val, dtype=mod1_val.dtype)
mod2_train = sc.AnnData(mod2_train, dtype=mod2_train.dtype)
mod2_val = sc.AnnData(mod2_val, dtype=mod2_val.dtype)

del train_mod1
del train_mod2

net = ContrastiveModel2(args)
logger.write('net1: ' + str(net) + '\n')

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 0}

# train model by contrastive
training_set = ModalityDataset2(mod1_train, mod2_train)
val_set = ModalityDataset2(mod1_val, mod2_val)

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

best_state_dict = train_contrastive(train_loader, val_loader, net, args, logger)

torch.save(best_state_dict,
           f'{param["save_model_path"]}{time_train}/model param contrastive.pkl')
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model param contrastive.pkl'))