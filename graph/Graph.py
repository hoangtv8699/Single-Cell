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
# pretrain_path = f'../pretrain/processed/{mod1_name}2{mod2_name}/'

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
train_mod1_ori = pk.load(open('../data/processed atac.pkl', 'rb'))
train_mod2_ori = sc.read_h5ad('../data/multiome/gex.h5ad')[:42492]
sc.pp.log1p(train_mod2_ori)
gene_list = os.listdir(f'{param["save_model_path"]}17_12_2022 17_49_55 atac embed')
gene_names = []
gene_dict = {}
for idx, gene in enumerate(gene_list):
    gene = gene.split('.')[0]
    gene_names.append(gene)
    gene_dict[gene] = idx

# pk.dump([gene_names, gene_dict], open(f'{param["save_model_path"]}/gene_names.pkl', 'wb'))

mod1 = train_mod1_ori
mod2 = train_mod2_ori[:, gene_names].X.toarray()
# src, dst, weights = pk.load(open('../data/graph.pkl', 'rb'))
src, dst, weights = pk.load(open('../data/pw_cosine.pkl', 'rb'))
edges = [src, dst]

logger.write('args: ' + str(args) + '\n')
# dump args
pk.dump(args, open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'wb'))

train_idx, val_idx = train_test_split(np.arange(42492), test_size=0.1, random_state=args.random_seed)
mod1_train = mod1[train_idx]
mod1_val = mod1[val_idx]
mod2_train = mod2[train_idx]
mod2_val = mod2[val_idx]

del train_mod1_ori
del train_mod2_ori

net = GCN([8, 1], [mod1_train.shape[1], 256, mod2_train.shape[1]])
logger.write('net: ' + str(net) + '\n')

training_set = GraphDataset(mod1_train, mod2_train)
val_set = GraphDataset(mod1_val, mod2_val)

best_state_dict = train(training_set, val_set, net, args, logger, edges, weights)
torch.save(best_state_dict,
           f'{param["save_model_path"]}{time_train}/GAT.pkl')
