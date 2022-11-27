import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import math


from utils import *

device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'

param = {
    'use_pretrained': True,
    'input_train_mod1': f'{dataset_path}test_mod2.h5ad',
    'input_train_mod2': f'{dataset_path}output.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}


mod1 = sc.read_h5ad(param['input_train_mod1']).X.toarray()
mod2 = sc.read_h5ad(param['input_train_mod2']).X.toarray()

rmse = mean_squared_error(mod1, mod2, squared=False)
print(rmse)