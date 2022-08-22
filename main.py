import pandas as pd
import scanpy as sc
import anndata as ad
import logging
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from utils import *

dataset_path = 'data/cite_BMMC_processed/'
pretrain_path= 'pretrain/cite_BMMC_processed/'

param = {
    'use_pretrained_data': True,
    'input_train_mod1': f'{dataset_path}mod1.h5ad',
    'input_train_mod2': f'{dataset_path}mod2.h5ad',
    'n_components_mod1': 100,
    'n_components_mod2': 100,
    'random_seed': 17,
    'output_pretrain': 'pretrain',
    'input_train_mod1_pretrained': f'{pretrain_path}mod1_train_svd.pkl',
    'input_train_mod2_pretrained': f'{pretrain_path}mod2_train_svd.pkl',
    'input_test_mod1_pretrained': f'{pretrain_path}mod1_test_svd.pkl',
    'input_test_mod2_pretrained': f'{pretrain_path}mod2_test_svd.pkl',
    'mod1_reducer_pretrained': f'{pretrain_path}svd_mod1.pkl',
    'mod2_reducer_pretrained': f'{pretrain_path}svd_mod2.pkl'
}


if not param['use_pretrained_data']:
    logging.info('Reading `h5ad` files...')
    train_mod1 = sc.read_h5ad(param['input_train_mod1'])
    mod1 = train_mod1.var['feature_types'][0]
    dataset_id = train_mod1.uns['dataset_id']
    input_train_mod1 = train_mod1.X

    train_mod2 = sc.read_h5ad(param['input_train_mod2'])
    var = train_mod2.var
    mod2 = train_mod2.var['feature_types'][0]
    input_train_mod2 = train_mod2.X

    mod1_train, mod1_test, mod2_train, mod2_test = train_test_split(input_train_mod1, input_train_mod2, test_size=0.1,
                                                                    random_state=param['random_seed'])

    mod1_train, mod1_reducer = embedding(mod1_train, param['n_components_mod1'], random_seed=param['random_seed'])
    mod2_train, mod2_reducer = embedding(mod2_train, param['n_components_mod1'], random_seed=param['random_seed'])

    mod1_test = mod1_reducer.transform(mod1_test)
    mod2_test = mod2_reducer.transform(mod2_test)

    pk.dump(mod1_train, open(param['input_train_mod1_pretrained'], "wb"))
    pk.dump(mod1_test, open(param['input_test_mod1_pretrained'], "wb"))
    pk.dump(mod2_train, open(param['input_train_mod2_pretrained'], "wb"))
    pk.dump(mod2_test, open(param['input_test_mod2_pretrained'], "wb"))
    pk.dump(mod1_reducer, open(param['mod1_reducer_pretrained'], "wb"))
    pk.dump(mod2_reducer, open(param['mod2_reducer_pretrained'], "wb"))
else:
    mod1_train = pk.load(open(param['input_train_mod1_pretrained'], "rb"))
    mod1_test = pk.load(open(param['input_test_mod1_pretrained'], "rb"))
    mod2_train = pk.load(open(param['input_train_mod2_pretrained'], "rb"))
    mod2_test = pk.load(open(param['input_test_mod2_pretrained'], "rb"))
    mod1_reducer = pk.load(open(param['mod1_reducer_pretrained'], "rb"))
    mod2_reducer = pk.load(open(param['mod2_reducer_pretrained'], "rb"))
