import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix

from utils import *

print(torch.cuda.is_available())
device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'
chr = 'chr1'

args = Namespace(
    random_seed=17,
    epochs=1000,
    lr=1e-4,
    patience=15,
    train_mod1=f'{dataset_path}{chr}/atac_train.h5ad',
    train_mod2=f'{dataset_path}{chr}/gex_train.h5ad',
    cajal_train_mod2=f'{dataset_path}train_mod2_chr1_cajal.h5ad',
    pretrain='../pretrain/',
    save_model_path='../saved_model/',
    logs_path='../logs/',
    weight_decay=0.8
)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y-%H_%M_%S") + f'-{mod1_name}2{mod2_name}-cajal-chr1'
os.mkdir(f'{args.save_model_path}{time_train}')
logger = open(f'{args.logs_path}{time_train}.log', 'a')

# get feature type
print('loading data')
mod1 = sc.read_h5ad(args.train_mod1)
mod2 = sc.read_h5ad(args.train_mod2)
cajal_mod2 = sc.read_h5ad(args.cajal_train_mod2)
cajal_mod2 = cajal_mod2[:, mod2.var_names]

# # normalize per batch
# train_batches = set(mod1.obs.batch)
# for batch in train_batches:
#     print(batch)
#     X_tmp = mod1[mod1.obs.batch == batch].X.toarray()
#     X_tmp = X_tmp.T
#     mean = np.mean(X_tmp, axis=1)
#     std = np.std(X_tmp, axis=1)
#     mean = mean.reshape(len(mean), 1)
#     std = std.reshape(len(std), 1)
#     X_tmp = (X_tmp - mean) / (std + 1)
#     X_tmp = X_tmp.T
#     mod1[mod1.obs.batch == batch].X = csr_matrix(X_tmp)

# norm
print('normalizing data')
X_train = mod1.X.toarray()
X_train = X_train.T
mean = np.mean(X_train, axis=1)
std = np.std(X_train, axis=1)
mean = mean.reshape(len(mean), 1)
std = std.reshape(len(std), 1)
info = {"means": mean, "sds": std}
with open(f'{args.save_model_path}{time_train}/transformation.pkl', "wb") as out:
    pk.dump(info, out)

X_train = (X_train - mean) / std
X_train = X_train.T
mod1.X = csr_matrix(X_train)

logger.write('args: ' + str(args) + '\n')

print('splitting data')
train_idx, val_idx = train_test_split(np.arange(mod1.shape[0]), test_size=0.1, random_state=args.random_seed)
mod1_train = mod1[train_idx]
mod1_val = mod1[val_idx]
mod2_train = mod2[train_idx]
mod2_val = mod2[val_idx]
cajal_mod2_train = cajal_mod2[train_idx]
cajal_mod2_val = cajal_mod2[val_idx]

net = ModalityNET(mod2_train.shape[1])
logger.write('net: ' + str(net) + '\n')

training_set = ModalityDataset(mod1_train.X.toarray(), cajal_mod2_train.X.toarray(), mod2_train.X.toarray())
val_set = ModalityDataset(mod1_val.X.toarray(), cajal_mod2_val.X.toarray(), mod2_val.X.toarray())

params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 0}

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

train_att(train_loader, val_loader, net, args, logger, time_train)

