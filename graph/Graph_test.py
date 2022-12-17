import pickle as pk

import numpy as np
import scanpy as sc
from scipy.sparse import csc_matrix
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

time_train = '26_11_2022 19_05_59 atac to gex'

# if load args
args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

# get feature type
test_mod1 = sc.read_h5ad(param['input_test_mod1'])
test_mod2 = sc.read_h5ad(param['input_test_mod2'])
src, dst, weights = pk.load(open('../data/graph.pkl', 'rb'))
edges = [src, dst]

params = {'batch_size': 256,
          'shuffle': False,
          'num_workers': 0}


mod1 = test_mod1.X.toarray()
mod2 = test_mod2.X.toarray()

test_set = GraphDataset(mod1, mod2)

net = GCN([1, 8, 4, 2], [2, 1])
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/GAT.pkl'))
net.cuda()
net.eval()

edges = torch.tensor(edges).cuda()
weights = torch.tensor(weights).float().cuda()

out_arr = []
for mod1, mod2 in test_set:
    mod1, mod2 = mod1.cuda(), mod2.cuda()
    out = net(mod1, edges, weights)
    out_arr.append(out.detach().cpu().numpy())

rmse = cal_rmse(csc_matrix(mod2), csc_matrix(out_arr))
print(rmse)