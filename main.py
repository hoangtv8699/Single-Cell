import pandas as pd
import scanpy as sc
import anndata as ad
import logging

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from utils import *


dataset_path = 'data/multiome_BMMC_processed/'

param = {
    'input_train_mod1': f'{dataset_path}mod1.h5ad',
    'input_train_mod2': f'{dataset_path}mod2.h5ad',
    'n_components_mod1': 100,
    'n_components_mod2': 100,
    'random_seed': 17,
    'output': 'output.h5ad'
}

logging.info('Reading `h5ad` files...')
train_mod1 = sc.read_h5ad(param['input_train_mod1'])
mod1 = train_mod1.var['feature_types'][0]
dataset_id = train_mod1.uns['dataset_id']
input_train_mod1 = train_mod1.X

# train_mod2 = sc.read_h5ad(param['input_train_mod2'])
# var = train_mod2.var
# mod2 = train_mod2.var['feature_types'][0]
# input_train_mod2 = train_mod2.X

input_train_mod1 = embedding(input_train_mod1, param['n_components_mod1'], random_seed=param['random_seed'])
# input_train_mod2 = embedding(input_train_mod2, param['n_components_mod1'], random_seed=param['random_seed'])

# mod1_train, mod1_test, mod2_train, mod2_test = train_test_split(input_train_mod1, input_train_mod2)

