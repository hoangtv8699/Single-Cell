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
    patience=15,
    train_mod1=f'{dataset_path}train_mod1.h5ad',
    train_mod2=f'{dataset_path}train_mod2.h5ad',
    train_mod2_cajal=f'{dataset_path}train_mod2_cajal.h5ad',
    pretrain='../pretrain/',
    save_model_path='../saved_model/',
    logs_path='../logs/',
    weight_decay=0.8
)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y-%H_%M_%S") + f'-{mod1_name}2{mod2_name}'
os.mkdir(f'{args.save_model_path}{time_train}')
logger = open(f'{args.logs_path}{time_train}.log', 'a')

gene_locus = pk.load(open('../craw/gene locus 2.pkl', 'rb'))
gene_dict = pk.load(open('../craw/gene infor 2.pkl', 'rb'))

# get feature type
print('loading data')
mod1_full = sc.read_h5ad(args.train_mod1)
mod2_full = sc.read_h5ad(args.train_mod2)
cajal_mod2_full = sc.read_h5ad(args.train_mod2_cajal)

for chr in range(1, 24):
    model_path = f'{args.save_model_path}{time_train}/chr{str(chr)}/model.pkl'
    tranformation_path = f'{args.save_model_path}{time_train}/chr{str(chr)}/transformation.pkl'
    mod1_path = f'{dataset_path}chr{str(chr)}/atac_train.h5ad'
    mod2_path = f'{dataset_path}chr{str(chr)}/gex_train.h5ad'
    # check if data folder created
    os.mkdir(f'{args.save_model_path}{time_train}/chr{str(chr)}')
    if not os.path.exists(f'{dataset_path}chr{str(chr)}'):
        os.mkdir(f'{dataset_path}chr{str(chr)}')
    # check if model trained
    if os.path.exists(model_path):
        print('model is trained: chr ', chr)
        logger.write('model is trained: chr ' + str(chr) + '\n')
        continue
    print('training chr: ', chr)
    logger.write('training chr: ' + str(chr) + '\n')
    # check if data created
    if os.path.exists(mod1_path) and os.path.exists(mod2_path):
        mod1 = sc.read_h5ad(mod1_path)
        mod2 = sc.read_h5ad(mod2_path)
    else:
        genes = []
        for key in gene_locus.keys():
            if int(gene_dict[key]['chromosome_name'][-3:]) == chr:
                genes.append(key)

        list_atac = []
        list_gene = []
        for atac in mod1_full.var_names:
            if atac.split('-')[0] == ('chr' + str(chr)):
                list_atac.append(atac)
        for gex in mod2_full.var_names:
            if gex in genes:
                list_gene.append(gex)

        mod1 = mod1_full[:, list_atac]
        mod2 = mod2_full[:, list_gene]

        mod1.write_h5ad(mod1_path)
        mod2.write_h5ad(mod2_path)
    print('number of atac: ', len(mod1.var_names))
    print('number of gene: ', len(mod2.var_names))
    logger.write('number of atac: ' + str(len(mod1.var_names)) + '\n')
    logger.write('number of gene: ' + str(len(mod2.var_names)) + '\n')
    #
    # cajal_mod2 = cajal_mod2_full[:, mod2.var_names]
    #
    # # column wise normalize
    # print('normalizing data')
    # X_train = mod1.X.toarray()
    # X_train = X_train.T
    # mean = np.mean(X_train, axis=1)
    # std = np.std(X_train, axis=1)
    # mean = mean.reshape(len(mean), 1)
    # std = std.reshape(len(std), 1)
    # info = {"means": mean, "sds": std}
    # with open(tranformation_path, "wb") as out:
    #     pk.dump(info, out)
    #
    # X_train = (X_train - mean) / std
    # X_train = X_train.T
    # mod1.X = csr_matrix(X_train)
    #
    # logger.write('args: ' + str(args) + '\n')
    #
    # print('splitting data')
    # train_idx, val_idx = train_test_split(np.arange(mod1.shape[0]), test_size=0.1, random_state=args.random_seed)
    # mod1_train = mod1[train_idx]
    # mod1_val = mod1[val_idx]
    # mod2_train = mod2[train_idx]
    # mod2_val = mod2[val_idx]
    # cajal_mod2_train = cajal_mod2[train_idx]
    # cajal_mod2_val = cajal_mod2[val_idx]
    #
    # net = ModalityNET(mod2_train.shape[1])
    # logger.write('net: ' + str(net) + '\n')
    #
    # training_set = ModalityDataset(mod1_train.X.toarray(), cajal_mod2_train.X.toarray(), mod2_train.X.toarray())
    # val_set = ModalityDataset(mod1_val.X.toarray(), cajal_mod2_val.X.toarray(), mod2_val.X.toarray())
    #
    # params = {'batch_size': 16,
    #           'shuffle': True,
    #           'num_workers': 0}
    #
    # train_loader = DataLoader(training_set, **params)
    # val_loader = DataLoader(val_set, **params)
    #
    # train_att(train_loader, val_loader, net, args, logger, time_train, model_path=model_path)

