import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


from utils import *

print(torch.cuda.is_available())
device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/{mod1_name}2{mod2_name}/'

args = Namespace(
    random_seed=17,
    epochs=1000,
    lr=1e-4,
    patience=15,
    train_mod1=f'{dataset_path}train_mod1.h5ad',
    train_mod2=f'{dataset_path}train_mod2.h5ad',
    save_model_path='../saved_model/',
    logs_path='../logs/',
    weight_decay=0.8
)

now = datetime.now()
# time_train = now.strftime("%d_%m_%Y-%H_%M_%S") + f'-{mod1_name}2{mod2_name}'
time_train = '14_02_2023-03_29_31-atac2gex'

if not os.path.exists(f'{args.save_model_path}{time_train}'):
    os.mkdir(f'{args.save_model_path}{time_train}')
logger = open(f'{args.logs_path}{time_train}.log', 'a')
logger.write('args: ' + str(args) + '\n')

# get feature type
print('loading data')
mod1 = sc.read_h5ad(args.train_mod1)
mod2 = sc.read_h5ad(args.train_mod2)

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
with open(f'{pretrain_path}transformation.pkl', "wb") as out:
    pk.dump(info, out)

for i in range(X_train.shape[0]):
    X_train[i] = (X_train[i] - mean[i]) / std[i]

print(X_train.shape)
# X_train = (X_train - mean) / std
X_train = X_train.T
# mod1.X = csr_matrix(X_train)

print('reducing data')
mod1_reducer = TruncatedSVD(n_components=128, random_state=17)
mod2_reducer = TruncatedSVD(n_components=128, random_state=17)
X = mod1_reducer.fit_transform(X_train)
y = mod2_reducer.fit_transform(mod2.X.toarray())
# save reducer model
with open(f'{pretrain_path}mod1_reducer.pkl', "wb") as out:
    pk.dump(mod1_reducer, out)
with open(f'{pretrain_path}mod2_reducer.pkl', "wb") as out:
    pk.dump(mod2_reducer, out)
# # if using saved pretrain
# with open(f'{pretrain_path}mod1_reducer.pkl', "rb") as f:
#     mod1_reducer = pk.load(f)
# with open(f'{pretrain_path}mod2_reducer.pkl', "rb") as f:
#     mod2_reducer = pk.load(f)
# X = mod1_reducer.transform(mod1.X)
# y = mod2_reducer.transform(mod2.X)

print('splitting data')
train_idx, val_idx = train_test_split(np.arange(mod1.shape[0]), test_size=0.1, random_state=args.random_seed)
mod1_train = X[train_idx]
mod1_val = X[val_idx]
mod2_train = y[train_idx]
mod2_val = y[val_idx]

net = SimpleNET(mod1_train.shape[1], mod2_train.shape[1])
logger.write('net: ' + str(net) + '\n')

training_set = ModalityDataset2(mod1_train, mod2_train)
val_set = ModalityDataset2(mod1_val, mod2_val)

params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 0}

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

train_simple(train_loader, val_loader, net, args, logger, time_train)

