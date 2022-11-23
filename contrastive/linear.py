import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import *

device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'

param = {
    'use_pretrained': True,
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}

args = Namespace(
    random_seed=17,
    epochs=1000,
    lr=1e-4,
    patience=10
)
now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + f' {mod1_name} to {mod2_name}'
# time_train = '03_10_2022 22_30_28 gex to atac'
os.mkdir(f'{param["save_model_path"]}{time_train}')
os.mkdir(f'{param["logs_path"]}{time_train}')

# get feature type
train_mod1_ori = sc.read_h5ad(param['input_train_mod1'])
train_mod2_ori = sc.read_h5ad(param['input_train_mod2'])

gene_locus = pk.load(open('../data/gene locus new.pkl', 'rb'))
gene_list = list(gene_locus.keys())
# gene_name = gene_list[0]

for gene_name in gene_list:
    train_mod1 = train_mod1_ori[:, gene_locus[gene_name]]
    train_mod2 = train_mod2_ori[:, gene_name]
    logger = open(f'{param["logs_path"]}{time_train}/{gene_name}.log', 'a')

    mod1 = train_mod1.X
    mod2 = train_mod2.X

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

    net = LinearRegressionModel(mod1.shape[1], mod2.shape[1])
    logger.write('net1: ' + str(net) + '\n')

    params = {'batch_size': 256,
              'shuffle': True,
              'num_workers': 0}

    # train model by contrastive
    training_set = ModalityDataset2(mod1_train, mod2_train)
    val_set = ModalityDataset2(mod1_val, mod2_val)

    train_loader = DataLoader(training_set, **params)
    val_loader = DataLoader(val_set, **params)

    best_state_dict = train_linear(train_loader, val_loader, net, args, logger)

    torch.save(best_state_dict,
               f'{param["save_model_path"]}{time_train}/{gene_name}.pkl')
