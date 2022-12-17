import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
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

# # if using raw data
# train_mod1_ori.X = normalize(train_mod1_ori.layers['counts'], axis=0)

# cell_type = 'CD4+ T activated'
# train_mod1_ori = train_mod1_ori[train_mod1_ori.obs['cell_type'] == cell_type, :]
# train_mod2_ori = train_mod2_ori[train_mod2_ori.obs['cell_type'] == cell_type, :]

gene_locus = pk.load(open('../craw/gene locus promoter.pkl', 'rb'))
gene_list = []
for gene_name in gene_locus.keys():
    if len(gene_locus[gene_name]) > 0:
        gene_list.append(gene_name)
# gene_name = gene_list[0]

# gene_list = ['AAK1']
# gene_locus['AAK1'] = ['chr2-69582036-69582843', 'chr2-69591589-69592502']

for gene_name in gene_list:
    train_mod1 = train_mod1_ori[:, gene_locus[gene_name]]
    train_mod2 = train_mod2_ori[:, gene_name]
    logger = open(f'{param["logs_path"]}{time_train}/{gene_name}.log', 'a')

    mod1 = train_mod1.X.toarray()
    mod2 = train_mod2.X.toarray()

    # train linear regression model
    logger.write('sklearn linear regression model' + '\n')
    net = LinearRegression()
    net.fit(mod1, mod2)
    pk.dump(net, open(f'{param["save_model_path"]}{time_train}/{gene_name}.pkl', 'wb'))
    print(gene_name)
    # break

    # # train torch model
    # logger.write('args1: ' + str(args) + '\n')
    # # dump args
    # pk.dump(args, open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'wb'))
    #
    # mod1_train, mod1_val, mod2_train, mod2_val = train_test_split(mod1, mod2, test_size=0.1, random_state=args.random_seed)
    #
    # mod1_train = sc.AnnData(mod1_train, dtype=mod1_train.dtype)
    # mod1_val = sc.AnnData(mod1_val, dtype=mod1_val.dtype)
    # mod2_train = sc.AnnData(mod2_train, dtype=mod2_train.dtype)
    # mod2_val = sc.AnnData(mod2_val, dtype=mod2_val.dtype)
    #
    # del train_mod1
    # del train_mod2
    #
    # net = LinearRegressionModel(mod1.shape[1], mod2.shape[1])
    # logger.write('net1: ' + str(net) + '\n')
    #
    # params = {'batch_size': 256,
    #           'shuffle': True,
    #           'num_workers': 0}
    #
    # training_set = ModalityDataset2(mod1_train, mod2_train)
    # val_set = ModalityDataset2(mod1_val, mod2_val)
    #
    # train_loader = DataLoader(training_set, **params)
    # val_loader = DataLoader(val_set, **params)
    #
    # best_state_dict = train_linear(train_loader, val_loader, net, args, logger)
    #
    # torch.save(best_state_dict,
    #            f'{param["save_model_path"]}{time_train}/{gene_name}.pkl')

