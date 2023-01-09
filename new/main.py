import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import *
print(torch.cuda.is_available())
device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/processed/{mod1_name}2{mod2_name}/'

args = Namespace(
    random_seed=17,
    epochs=1000,
    lr=1e-4,
    patience=10,
    train_mod1=f'{dataset_path}atac_train_chr1.h5ad',
    train_mod2=f'{dataset_path}gex_train_chr1.h5ad',
    pretrain='../pretrain/',
    save_model_path='../saved_model/',
    logs_path='../logs/'
)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y-%H_%M_%S") + f'-{mod1_name}2{mod2_name}'
os.mkdir(f'{args.save_model_path}{time_train}')
os.mkdir(f'{args.logs_path}{time_train}')
logger = open(f'{args.logs_path}{time_train}.log', 'a')

# get feature type
mod1 = sc.read_h5ad(args.train_mod1)
mod2 = sc.read_h5ad(args.train_mod2)

logger.write('args: ' + str(args) + '\n')
# dump args
pk.dump(args, open(f'{args.save_model_path}{time_train}/args net.pkl', 'wb'))

train_idx, val_idx = train_test_split(np.arange(mod1.shape[0]), test_size=0.1, random_state=args.random_seed)
mod1_train = mod1[train_idx]
mod1_val = mod1[val_idx]
mod2_train = mod2[train_idx]
mod2_val = mod2[val_idx]

net = ModalityNET([mod1_train.shape[1], 256, mod2_train.shape[1]])
logger.write('net: ' + str(net) + '\n')

training_set = ModalityDataset(mod1_train, mod2_train)
val_set = ModalityDataset(mod1_val, mod2_val)

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 0}

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

best_state_dict = train(train_loader, val_loader, net, args, logger)
torch.save(best_state_dict,
           f'{args.save_model_path}{time_train}/GAT.pkl')
