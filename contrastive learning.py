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

from datetime import datetime

from utils import *

device = torch.device("cuda:0")

dataset_path = 'data/train/multiome/'
pretrain_path = 'pretrain/'

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

args1 = Namespace(
    input_feats=13431,
    num_class=22,
    embed_hid_feats=512,
    latent_feats=32,
    class_hid_feats=512,
    pred_hid_feats=512,
    out_feats=13431,
    random_seed=17,
    activation='relu',
    act_out='sigmoid',
    num_embed_layer=4,  # if using residual, this number must divisible by 2
    num_class_layer=4,  # if using residual, this number must divisible by 2
    num_pred_layer=4,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=1000,
    lr_classification=1e-4,
    lr_embed=1e-4,
    lr_predict=1e-3,
    normalization='batch',
    patience=10
)

args2 = Namespace(
    input_feats=256,
    num_class=22,
    embed_hid_feats=1024,
    latent_feats=1024,
    class_hid_feats=1024,
    pred_hid_feats=1024,
    out_feats=13431,
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

# get feature type
logging.info('Reading `h5ad` files...')
train_mod1 = sc.read_h5ad(param['input_train_mod1'])
train_mod2 = sc.read_h5ad(param['input_train_mod2'])
mod1 = train_mod1.var['feature_types'][0]
mod2 = train_mod2.var['feature_types'][0]

# get input and encode label for classification training
LE = LabelEncoder()
train_mod1.obs["class_label"] = LE.fit_transform(train_mod1.obs["cell_type"])
input_label = train_mod1.obs["class_label"].to_numpy()
logger.write('class name: ' + str(LE.classes_) + '\n')

# select feature
if param['use_pretrained']:
    subset1 = pk.load(open(param['subset_pretrain1'], 'rb'))
    subset2 = pk.load(open(param['subset_pretrain2'], 'rb'))
else:
    subset1 = analysis_features(train_mod1, top=100)
    subset2 = analysis_features(train_mod2, top=1000)
    pk.dump(subset1, open(param['subset_pretrain1'], 'wb'))
    pk.dump(subset2, open(param['subset_pretrain2'], 'wb'))
    logger.write(f'selected {mod1} feature: ' + str(subset1) + '\n')
    logger.write(f'selected {mod1} feature: ' + str(subset1) + '\n')

input_train_mod1 = train_mod1.X
input_train_mod2 = train_mod2.X
# # if using reduced data
# input_train_mod1_reduced = train_mod1[:, list(subset1)].X
# input_train_mod2_reduced = train_mod2[:, list(subset2)].X
# if using full data
input_train_mod1_reduced = train_mod1.X
input_train_mod2_reduced = train_mod2.X
# if using raw data instead of processed data
# input_train_mod1_reduced = train_mod1[:, list(subset1)].layers['counts']
# input_train_mod2_reduced = train_mod2[:, list(subset2)].layers['counts']

args1.classes_ = LE.classes_
args1.num_class = len(args1.classes_)
args1.input_feats = input_train_mod1_reduced.shape[1]
args1.out_feats = input_train_mod2.shape[1]
args2.classes_ = LE.classes_
args2.num_class = len(args1.classes_)
args2.input_feats = input_train_mod2_reduced.shape[1]
args2.out_feats = input_train_mod1.shape[1]

logger.write('args1: ' + str(args1) + '\n')
logger.write('args2: ' + str(args2) + '\n')
# dump args
pk.dump(args1, open(f'{param["save_model_path"]}{time_train}/args net1.pkl', 'wb'))
pk.dump(args2, open(f'{param["save_model_path"]}{time_train}/args net2.pkl', 'wb'))

mod1_train, mod1_val, mod2_train, mod2_val, label_train, label_val, mod1_reduced_train, mod1_reduced_val, \
mod2_reduced_train, mod2_reduced_val = train_test_split(input_train_mod1,
                                                        input_train_mod2, input_label,
                                                        input_train_mod1_reduced,
                                                        input_train_mod2_reduced,
                                                        test_size=0.1,
                                                        random_state=args1.random_seed,
                                                        stratify=input_label)

mod1_train = sc.AnnData(mod1_train, dtype=mod1_train.dtype)
mod1_val = sc.AnnData(mod1_val, dtype=mod1_val.dtype)
mod2_train = sc.AnnData(mod2_train, dtype=mod2_train.dtype)
mod2_val = sc.AnnData(mod2_val, dtype=mod2_val.dtype)
mod1_reduced_train = sc.AnnData(mod1_reduced_train, dtype=mod1_reduced_train.dtype)
mod1_reduced_val = sc.AnnData(mod1_reduced_val, dtype=mod1_reduced_val.dtype)
mod2_reduced_train = sc.AnnData(mod2_reduced_train, dtype=mod2_reduced_train.dtype)
mod2_reduced_val = sc.AnnData(mod2_reduced_val, dtype=mod2_reduced_val.dtype)

del train_mod1
del train_mod2
del input_train_mod1
del input_train_mod2
del input_train_mod1_reduced
del input_train_mod2_reduced

net1 = ContrastiveModel(args1)
net2 = ContrastiveModel(args2)
logger.write('net1: ' + str(net1) + '\n')
logger.write('net2: ' + str(net2) + '\n')

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 0}

# # train model to classification
# training_set1 = ModalityDataset2(mod1_reduced_train, label_train, types='classification')
# training_set2 = ModalityDataset2(mod2_reduced_train, label_train, types='classification')
# val_set1 = ModalityDataset2(mod1_reduced_val, label_val, types='classification')
# val_set2 = ModalityDataset2(mod2_reduced_val, label_val, types='classification')
#
# train_loader1 = DataLoader(training_set1, **params)
# train_loader2 = DataLoader(training_set2, **params)
# val_loader1 = DataLoader(val_set1, **params)
# val_loader2 = DataLoader(val_set2, **params)
#
# best_state_dict1 = train_classification(train_loader1, val_loader1, net1, args1, logger)
# torch.save(best_state_dict1,
#            f'{param["save_model_path"]}{time_train}/model {mod1} param classification.pkl')
#
# best_state_dict2 = train_classification(train_loader2, val_loader2, net2, args2, logger)
# torch.save(best_state_dict2,
#            f'{param["save_model_path"]}{time_train}/model {mod2} param classification.pkl')
#
# # load pretrained from dir
# net1.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod1} param classification.pkl'))
# net2.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod2} param classification.pkl'))
#
# # train model by contrastive
# training_set = ModalityDataset2(mod1_reduced_train, mod2_reduced_train, types='2mod')
# val_set = ModalityDataset2(mod1_reduced_val, mod2_reduced_val, types='2mod')
#
# train_loader = DataLoader(training_set, **params)
# val_loader = DataLoader(val_set, **params)
#
# best_state_dict1, best_state_dict2 = train_contrastive(train_loader, val_loader, net1, net2, args1, logger)
#
# torch.save(best_state_dict1,
#            f'{param["save_model_path"]}{time_train}/model {mod1} param contrastive.pkl')
# torch.save(best_state_dict2,
#            f'{param["save_model_path"]}{time_train}/model {mod2} param contrastive.pkl')
#
# # load pretrained from dir
# net1.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod1} param contrastive.pkl'))
# net2.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod2} param contrastive.pkl'))

# mod2_reduced_train, mod2_reducer = embedding(mod2_reduced_train, args2.input_feats, random_seed=args2.random_seed)

# train model to predict modality
training_set1 = ModalityDataset2(mod1_reduced_train, mod2_train, types='2mod')
training_set2 = ModalityDataset2(mod2_reduced_train, mod1_train, types='2mod')
val_set1 = ModalityDataset2(mod1_reduced_val, mod2_val, types='2mod')
val_set2 = ModalityDataset2(mod2_reduced_val, mod1_val, types='2mod')

train_loader1 = DataLoader(training_set1, **params)
train_loader2 = DataLoader(training_set2, **params)
val_loader1 = DataLoader(val_set1, **params)
val_loader2 = DataLoader(val_set2, **params)

# best_state_dict1 = train_predict(train_loader1, val_loader1, net1, args1, logger)
# torch.save(best_state_dict1,
#            f'{param["save_model_path"]}{time_train}/model {mod1} param predict.pkl')

best_state_dict2 = train_predict(train_loader2, val_loader2, net2, args2, logger)
torch.save(best_state_dict2,
           f'{param["save_model_path"]}{time_train}/model {mod2} param predict.pkl')
