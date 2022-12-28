import pickle as pk
import os
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error


from utils import *

# device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'

dataset_path = f'../data/processed/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/processed/{mod1_name}2{mod2_name}/'

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}

time_train = '28_12_2022-13_31_44-atac2gex'

# if load args
args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

# get feature type
test_mod1 = pk.load(open('../data/processed atac test.pkl', 'rb'))
test_mod2 = sc.read_h5ad('../data/multiome/gex.h5ad')[42492:43492]
sc.pp.log1p(test_mod2)
cajal = sc.read_h5ad('../data/paper data/atac2gex/output.h5ad')
gex_paper = sc.read_h5ad('../data/paper data/atac2gex/test_mod2.h5ad')
gene_list = os.listdir(f'{param["save_model_path"]}17_12_2022 17_49_55 atac embed')
gene_names = []
gene_dict = {}
for idx, gene in enumerate(gene_list):
    gene = gene.split('.')[0]
    gene_names.append(gene)
    gene_dict[gene] = idx


# src, dst, weights = pk.load(open('../data/graph.pkl', 'rb'))
src, dst, weights = pk.load(open('../data/pw_cosine.pkl', 'rb'))
edges = [src, dst]

params = {'batch_size': 256,
          'shuffle': False,
          'num_workers': 0}


mod1 = test_mod1
mod2 = test_mod2[:, gene_names].X.toarray()
cajal = cajal[:, gene_names].X
gex_paper = gex_paper[:, gene_names].X

test_set = GraphDataset(mod1, mod2)

net = GCN([8, 1], [mod1.shape[1], 256, mod2.shape[1]])
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/GAT.pkl'))
net.cuda()
net.eval()

edges = torch.tensor(edges).cuda()
# weights = torch.tensor(weights).float().cuda()

out_arr = []
for mod1, mod2 in test_set:
    mod1, mod2 = mod1.cuda(), mod2.cuda()
    out = net(mod1, edges)
    out_arr.append(out.detach().cpu().numpy())

rmse = cal_rmse(test_mod2[:, gene_names].X, csr_matrix(out_arr))
rmse2 = cal_rmse(gex_paper, cajal)
print(rmse, rmse2)