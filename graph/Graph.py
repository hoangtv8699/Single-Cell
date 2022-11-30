import logging
import os
import scanpy as sc
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
dataset_path = f'../data/processed/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/processed/{mod1_name}2{mod2_name}/'

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
time_train = now.strftime("%d_%m_%Y-%H_%M_%S") + f'-{mod1_name}2{mod2_name}'
# time_train = '03_10_2022 22_30_28 gex to atac'
os.mkdir(f'{param["save_model_path"]}{time_train}')
os.mkdir(f'{param["logs_path"]}{time_train}')
logger = open(f'{param["logs_path"]}{time_train}/gat.log', 'a')

# get feature type
train_mod1_ori = sc.read_h5ad(param['input_train_mod1'])
train_mod2_ori = sc.read_h5ad(param['input_train_mod2'])
mod1 = train_mod1_ori.X.toarray()
mod2 = train_mod2_ori.X.toarray()
src, dst, weights = pk.load(open('../data/graph.pkl', 'rb'))
edges = [src, dst]

logger.write('args: ' + str(args) + '\n')
# dump args
pk.dump(args, open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'wb'))

mod1_train, mod1_val, mod2_train, mod2_val = train_test_split(mod1, mod2, test_size=0.1, random_state=args.random_seed)
del train_mod1_ori
del train_mod2_ori

net = GCN([1, 8, 4, 2], [2, 1])
logger.write('net: ' + str(net) + '\n')
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}

training_set = GraphDataset(mod1_train, mod2_train)
val_set = GraphDataset(mod1_val, mod2_val)

best_state_dict = train(training_set, val_set, net, args, logger, edges, weights)
torch.save(best_state_dict,
           f'{param["save_model_path"]}{time_train}/GAT.pkl')
