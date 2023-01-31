from sklearn.decomposition import TruncatedSVD
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

import scanpy as sc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import *

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
# pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'

# args = Namespace(
#     random_seed=17,
#     epochs=1000,
#     lr=1e-4,
#     patience=10,
#     train_mod1=f'{dataset_path}atac_train_chr1.h5ad',
#     train_mod2=f'{dataset_path}gex_train_chr1.h5ad',
#     save_model_path='../saved_model/',
#     logs_path='../logs/'
# )

# mod1 = sc.read_h5ad(args.train_mod1)

# train_total = np.sum(mod1.X.toarray(), axis=1)
# train_batches = set(mod1.obs.batch)
# train_batches_dict = {}
# for batch in train_batches:
#     train_batches_dict[batch] = {}
#     train_batches_dict[batch]['mean'] = np.mean(train_total[mod1.obs.batch == batch])
#     train_batches_dict[batch]['std'] = np.std(train_total[mod1.obs.batch == batch])

# mod1.X = mod1.X.toarray()
# for i in range(mod1.X.shape[0]):
#     mod1.X[i] = (mod1.X[i] - train_batches_dict[mod1.obs.batch[i]]['mean'])/train_batches_dict[mod1.obs.batch[i]]['std']


# svd = TruncatedSVD(n_components=64, n_iter=7, random_state=42)
# svd.fit(mod1.X)

# pk.dump(svd, open(f'{pretrain_path}atac 64.pkl', 'wb'))

src, dst, weight = pk.load(open(f'{dataset_path}pw copy.pkl', 'rb'))
print(np.unique(src))

