import os
import time
import dgl
import torch
import pandas as pd
import scanpy as sc
import anndata as ad
import logging
import pickle as pk
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from argparse import Namespace
from torch.utils.data import DataLoader
from scipy.sparse import csc_matrix
from datetime import datetime

from utils import *

device = torch.device("cuda:0")

dataset_path = 'data/train/multiome/'
pretrain_path = 'pretrain/'


def calculate_rmse(true_test_mod2, pred_test_mod2):
    if pred_test_mod2.var["feature_types"][0] == "GEX":
        return mean_squared_error(true_test_mod2.layers["log_norm"].toarray(), pred_test_mod2.X, squared=False)
    else:
        raise NotImplementedError("Only set up to calculate RMSE for GEX data")


param = {
    'use_pretrained': True,
    'input_train_mod1': f'{dataset_path}gex.h5ad',
    'input_train_mod2': f'{dataset_path}atac.h5ad',
    'subset_pretrain1': f'{pretrain_path}GEX_subset.pkl',
    'subset_pretrain2': f'{pretrain_path}ATAC_subset.pkl',
    'output_pretrain': 'pretrain/',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}


def baseline_linear(input_train_mod1, input_train_mod2, input_test_mod1):
    '''Baseline method training a linear regressor on the input data'''
    input_mod1 = ad.concat(
        {"train": input_train_mod1, "test": input_test_mod1},
        axis=0,
        join="outer",
        label="group",
        fill_value=0,
        index_unique="-",
    )

    # Binarize ATAC
    if input_train_mod1.var["feature_types"][0] == "ATAC":
        input_mod1.X[input_mod1.X > 1] = 1
    elif input_train_mod2.var["feature_types"][0] == "ATAC":
        input_train_mod2.X[input_mod1.X > 1] = 1

    # Do PCA on the input data
    logging.info('Performing dimensionality reduction on modality 1 values...')
    embedder_mod1 = TruncatedSVD(n_components=50)
    mod1_pca = embedder_mod1.fit_transform(input_mod1.X)

    logging.info('Performing dimensionality reduction on modality 2 values...')
    embedder_mod2 = TruncatedSVD(n_components=50)
    mod2_pca = embedder_mod2.fit_transform(input_train_mod2.layers["log_norm"])

    # split dimred mod 1 back up for training
    X_train = mod1_pca[input_mod1.obs['group'] == 'train']
    X_test = mod1_pca[input_mod1.obs['group'] == 'test']
    y_train = mod2_pca

    assert len(X_train) + len(X_test) == len(mod1_pca)

    logging.info('Running Linear regression...')

    reg = LinearRegression()

    # Train the model on the PCA reduced modality 1 and 2 data
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    # Project the predictions back to the modality 2 feature space
    y_pred = y_pred @ embedder_mod2.components_

    pred_test_mod2 = ad.AnnData(
        X=y_pred,
        obs=input_test_mod1.obs,
        var=input_train_mod2.var,

    )

    # Add the name of the method to the result
    pred_test_mod2.uns["method"] = "linear"

    return pred_test_mod2


args2 = Namespace(
    input_feats=256,
    num_class=22,
    embed_hid_feats=512,
    latent_feats=128,
    class_hid_feats=512,
    pred_hid_feats=512,
    out_feats=256,
    random_seed=17,
    activation='relu',
    act_out='relu',
    num_embed_layer=4,  # if using residual, this number must divisible by 2
    num_class_layer=4,  # if using residual, this number must divisible by 2
    num_pred_layer=4,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=1000,
    lr_classification=1e-4,
    lr_embed=1e-4,
    lr_predict=1e-4,
    normalization='batch',
    patience=10
)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S")
# time_train = '26_09_2022 02_08_15'
logger = open(f'{param["logs_path"]}{time_train}.log', 'w')
os.mkdir(f'{param["save_model_path"]}{time_train}')

train_mod1 = sc.read_h5ad(param['input_train_mod1'])
train_mod2 = sc.read_h5ad(param['input_train_mod2'])
mod1 = train_mod1.var['feature_types'][0]
mod2 = train_mod2.var['feature_types'][0]

mod1_reducer = pk.load(open('pretrain/GEX reducer multiome.pkl', "rb"))
mod2_reducer = pk.load(open('pretrain/ATAC reducer multiome.pkl', "rb"))

input_train_mod1_reduced = csc_matrix(mod1_reducer.transform(train_mod1.X))
input_train_mod2_reduced = csc_matrix(mod2_reducer.transform(train_mod2.X))

mod1_reduced_train, mod1_reduced_val, mod2_reduced_train, mod2_reduced_val = train_test_split(input_train_mod1_reduced,
                                                                                              input_train_mod2_reduced,
                                                                                              test_size=0.1,
                                                                                              random_state=17)

mod1_reduced_train = sc.AnnData(mod1_reduced_train, dtype=mod1_reduced_train.dtype)
mod1_reduced_val = sc.AnnData(mod1_reduced_val, dtype=mod1_reduced_val.dtype)
mod2_reduced_train = sc.AnnData(mod2_reduced_train, dtype=mod2_reduced_train.dtype)
mod2_reduced_val = sc.AnnData(mod2_reduced_val, dtype=mod2_reduced_val.dtype)

# train model to predict modality
training_set2 = ModalityDataset2(mod2_reduced_train, mod1_reduced_train, types='2mod')
val_set2 = ModalityDataset2(mod2_reduced_val, mod1_reduced_val, types='2mod')

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 0}

net2 = LinearRegressionModel(256, 256)

train_loader2 = DataLoader(training_set2, **params)
val_loader2 = DataLoader(val_set2, **params)

best_state_dict2 = train_predict(train_loader2, val_loader2, net2, args2, logger, mod1_reducer)
torch.save(best_state_dict2,
           f'{param["save_model_path"]}{time_train}/model {mod2} param predict.pkl')
