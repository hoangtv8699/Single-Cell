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

args = Namespace(
    random_seed=17,
    epochs=1000,
    lr=1e-4,
    patience=10,
    train_mod1=f'{dataset_path}atac_train_chr1_2.h5ad',
    train_mod2=f'{dataset_path}gex_train_chr1_2.h5ad',
    train_mod1_domain=f'{dataset_path}atac_train_chr1.h5ad',
    test_mod1= f'{dataset_path}atac_test_chr1_2.h5ad',
    test_mod2= f'{dataset_path}gex_test_chr1_2.h5ad',
    test_mod1_domain=f'{dataset_path}atac_test_chr1.h5ad',
    pretrain='../pretrain/',
    save_model_path='../saved_model/',
    logs_path='../logs/',
    weight_decay=0.8
)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y-%H_%M_%S") + f'-{mod1_name}2{mod2_name}'
os.mkdir(f'{args.save_model_path}{time_train}')
# os.mkdir(f'{args.logs_path}{time_train}')
logger = open(f'{args.logs_path}{time_train}.log', 'a')

gene_locus = pk.load(open('../craw/gene locus 2.pkl', 'rb'))
gene_dict = pk.load(open('../craw/gene infor 2.pkl', 'rb'))


i = 0
list_gene = []
list_atac = []
for key in gene_locus.keys():
    if len(gene_locus[key]) > 30 and int(gene_dict[key]['chromosome_name'][-3:]) == 1:
        list_gene.append(key)
        list_atac += gene_locus[key]
        i += 1
        if i > 0:
            break

print(len(list_gene))
print(len(list_atac))

# get feature type
mod1 = sc.read_h5ad(args.train_mod1)
mod2 = sc.read_h5ad(args.train_mod2)[:,list_gene]
# mod1_domain = sc.read_h5ad(args.train_mod1_domain)
mod1_domain = mod1[:, list_atac]
mod1_test = sc.read_h5ad(args.test_mod1)
mod2_test = sc.read_h5ad(args.test_mod2)[:,list_gene]
# mod1_domain_test = sc.read_h5ad(args.test_mod1_domain)
mod1_domain_test = mod1_test[:, list_atac]

# svd = pk.load(open(f'{pretrain_path}atac 64.pkl', 'rb'))

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

# mod1.X = csr_matrix(mod1.X)
# # mod1.X = svd.transform(mod1.X)
# mod1 = sc.AnnData(
#     X=svd.transform(mod1.X)
# )

logger.write('args: ' + str(args) + '\n')
# dump args
pk.dump(args, open(f'{args.save_model_path}{time_train}/args net.pkl', 'wb'))

train_idx, val_idx = train_test_split(np.arange(mod1.shape[0]), test_size=0.1, random_state=args.random_seed)
mod1_train = mod1[train_idx]
mod1_val = mod1[val_idx]
mod1_domain_train = mod1_domain[train_idx]
mod1_domain_val = mod1_domain[val_idx]
mod2_train = mod2[train_idx]
mod2_val = mod2[val_idx]

net = AutoEncoder(mod1_train.shape[1], mod1_domain_train.shape[1], mod2_train.shape[1])
logger.write('net: ' + str(net) + '\n')

training_set = ModalityDataset2(mod1_train, mod1_domain_train, mod2_train)
val_set = ModalityDataset2(mod1_val, mod1_domain_val, mod2_val)
test_set = ModalityDataset2(mod1_test, mod1_domain_test, mod2_test)

params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 0}

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)
test_loader = DataLoader(val_set, **params)

best_state_dict = train_att(train_loader, val_loader, test_loader, net, args, logger)
torch.save(best_state_dict,
           f'{args.save_model_path}{time_train}/GAT.pkl')
