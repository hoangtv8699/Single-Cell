import pickle as pk
import os
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from argparse import Namespace

from utils import *

# device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'

dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'

args = Namespace(
    random_seed=17,
    epochs=1000,
    lr=1e-4,
    patience=15,
    test_mod1=f'{dataset_path}test_mod1.h5ad',
    test_mod2=f'{dataset_path}test_mod2.h5ad',
    test_mod2_cajal=f'{dataset_path}test_mod2_simple.h5ad',
    pretrain='../pretrain/',
    save_model_path='../saved_model/',
    logs_path='../logs/',
    weight_decay=0.8
)

# 0.217/0.2152
# time_train = '10_01_2023-15_30_18-atac2gex'
# 0.216/0.2152
# time_train = '11_01_2023-14_56_52-atac2gex'
time_train = '22_04_2023-15_40_47-atac2gex-23chr'

gene_locus = pk.load(open('../craw/gene locus 2.pkl', 'rb'))
gene_dict = pk.load(open('../craw/gene infor 2.pkl', 'rb'))

# get feature type
mod1_full = sc.read_h5ad(args.test_mod1)
mod2_full = sc.read_h5ad(args.test_mod2)

cajal_full = []
label_full = []
out_full = []

for chr in range(1, 23):
    model_path = f'{args.save_model_path}{time_train}/chr{str(chr)}/model.pkl'
    tranformation_path = f'{args.save_model_path}{time_train}/chr{str(chr)}/transformation.pkl'
    mod1_path = f'{dataset_path}chr{str(chr)}/atac_test.h5ad'
    mod2_path = f'{dataset_path}chr{str(chr)}/gex_test.h5ad'
    # check if model trained
    if not os.path.exists(model_path):
        print('model is not trained: chr ', chr)
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

    with open(tranformation_path, "rb") as f:
        info = pk.load(f)

    X_test = mod1.X.toarray()
    X_test = X_test.T
    X_test = (X_test - info["means"]) / info["sds"]
    X_test = X_test.T
    mod1.X = csr_matrix(X_test)

    cajal = sc.read_h5ad('../data/paper data/atac2gex/test_mod2_simple.h5ad')
    cajal_out = cajal[:, mod2.var_names]

    params = {'batch_size': 16,
              'shuffle': False,
              'num_workers': 0}
    test_set = ModalityDataset(mod1.X.toarray(), cajal_out.X.toarray(), mod2.X.toarray())
    test_loader = DataLoader(test_set, **params)

    net = torch.load(model_path)
    net.cuda()
    net.eval()

    outs = []
    for mod1_batch, mod1_domain_batch, mod2_batch in test_loader:
        mod1_batch, mod1_domain_batch, mod2_batch = mod1_batch.cuda(), mod1_domain_batch.cuda(), mod2_batch.cuda()
        out = net(mod1_batch, mod1_domain_batch)
        if len(outs) == 0:
            outs = out.detach().cpu().numpy()
        else:
            outs = np.concatenate((outs, out.detach().cpu().numpy()), axis=0)


    rmse = cal_rmse(mod2.X, csr_matrix(outs))
    rmse2 = cal_rmse(mod2.X, cajal_out.X)
    print('result on chr', chr, ': ' ,rmse, rmse2)

    if len(cajal_full) == 0:
        cajal_full = cajal_out.X.toarray()
        label_full = mod2.X.toarray()
        out_full = outs
    else:
        cajal_full = np.concatenate((cajal_full, cajal_out.X.toarray()), axis=1)
        label_full = np.concatenate((label_full, mod2.X.toarray()), axis=1)
        out_full = np.concatenate((out_full, outs), axis=1)

rmse_full = cal_rmse(csr_matrix(label_full), csr_matrix(out_full))
rmse2_full = cal_rmse(csr_matrix(label_full), csr_matrix(cajal_full))
print(label_full.shape)
print('result on full:' ,rmse_full, rmse2_full)
