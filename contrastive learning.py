import os
import time
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

param = {
    'use_pretrained': True,
    'input_train_mod1': f'{dataset_path}gex.h5ad',
    'input_train_mod2': f'{dataset_path}atac.h5ad',
    'subset_pretrain1': f'{pretrain_path}GEX reducer multiome nolog.pkl',
    'subset_pretrain2': f'{pretrain_path}ATAC reducer multiome nolog.pkl',
    'output_pretrain': 'pretrain/',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}

args1 = Namespace(
    input_feats=256,
    num_class=22,
    embed_hid_feats=512,
    latent_feats=64,
    class_hid_feats=512,
    pred_hid_feats=512,
    out_feats=256,
    random_seed=17,
    activation='relu',
    act_out='one',
    num_embed_layer=8,  # if using residual, this number must divisible by 2
    num_class_layer=6,  # if using residual, this number must divisible by 2
    num_pred_layer=6,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=1000,
    lr_classification=1e-4,
    lr_embed=1e-4,
    lr_predict=1e-4,
    normalization='batch',
    patience=10
)

args2 = Namespace(
    input_feats=256,
    num_class=22,
    embed_hid_feats=512,
    latent_feats=64,
    class_hid_feats=512,
    pred_hid_feats=512,
    out_feats=256,
    random_seed=17,
    activation='relu',
    act_out='none',
    num_embed_layer=8,  # if using residual, this number must divisible by 2
    num_class_layer=6,  # if using residual, this number must divisible by 2
    num_pred_layer=6,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=1000,
    lr_classification=1e-4,
    lr_embed=1e-4,
    lr_predict=1e-4,
    normalization='batch',
    patience=10
)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + 'no act_out'
# time_train = '27_09_2022 09_15_57 mod'
os.mkdir(f'{param["save_model_path"]}{time_train}')
logger = open(f'{param["logs_path"]}{time_train}.log', 'a')

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

mod1_reducer = pk.load(open(param['subset_pretrain1'], 'rb'))
mod2_reducer = pk.load(open(param['subset_pretrain2'], 'rb'))

# log norm train mod1
sc.pp.log1p(train_mod1)

# net1 input and output
net1_input = csc_matrix(mod1_reducer.transform(train_mod1.X))
net1_output = csc_matrix(mod2_reducer.transform(train_mod2.X))
net2_input = csc_matrix(mod2_reducer.transform(train_mod2.X))
net2_output = csc_matrix(mod1_reducer.transform(train_mod1.X))

# # if not using reducer
# net1_input = train_mod1.X
# net1_output = train_mod2.X
# net2_input = train_mod2.X
# net2_output = train_mod1.X

args1.classes_ = LE.classes_
args1.num_class = len(args1.classes_)
args1.input_feats = net1_input.shape[1]
args1.out_feats = net1_output.shape[1]
args2.classes_ = LE.classes_
args2.num_class = len(args2.classes_)
args2.input_feats = net2_input.shape[1]
args2.out_feats = net2_output.shape[1]

logger.write('args1: ' + str(args1) + '\n')
logger.write('args2: ' + str(args2) + '\n')
# dump args
pk.dump(args1, open(f'{param["save_model_path"]}{time_train}/args net1.pkl', 'wb'))
pk.dump(args2, open(f'{param["save_model_path"]}{time_train}/args net2.pkl', 'wb'))

net1_input_train, net1_input_val, net1_output_train, net1_output_val, \
net2_input_train, net2_input_val, net2_output_train, net2_output_val, \
label_train, label_val = train_test_split(net1_input,
                                          net1_output,
                                          net2_input,
                                          net2_output,
                                          input_label,
                                          test_size=0.1,
                                          random_state=args1.random_seed,
                                          stratify=input_label)

net1_input_train = sc.AnnData(net1_input_train, dtype=net1_input_train.dtype)
net1_input_val = sc.AnnData(net1_input_val, dtype=net1_input_val.dtype)
net1_output_train = sc.AnnData(net1_output_train, dtype=net1_output_train.dtype)
net1_output_val = sc.AnnData(net1_output_val, dtype=net1_output_val.dtype)
net2_input_train = sc.AnnData(net2_input_train, dtype=net2_input_train.dtype)
net2_input_val = sc.AnnData(net2_input_val, dtype=net2_input_val.dtype)
net2_output_train = sc.AnnData(net2_output_train, dtype=net2_output_train.dtype)
net2_output_val = sc.AnnData(net2_output_val, dtype=net2_output_val.dtype)

del train_mod1
del train_mod2

net1 = ContrastiveModel(args1)
net2 = ContrastiveModel(args2)
logger.write('net1: ' + str(net1) + '\n')
logger.write('net2: ' + str(net2) + '\n')

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 0}

# train model to classification
training_set1 = ModalityDataset2(net1_input_train, label_train, types='classification')
training_set2 = ModalityDataset2(net2_input_train, label_train, types='classification')
val_set1 = ModalityDataset2(net1_input_val, label_val, types='classification')
val_set2 = ModalityDataset2(net2_input_val, label_val, types='classification')

train_loader1 = DataLoader(training_set1, **params)
train_loader2 = DataLoader(training_set2, **params)
val_loader1 = DataLoader(val_set1, **params)
val_loader2 = DataLoader(val_set2, **params)

best_state_dict1 = train_classification(train_loader1, val_loader1, net1, args1, logger)
torch.save(best_state_dict1,
           f'{param["save_model_path"]}{time_train}/model {mod1} param classification.pkl')

best_state_dict2 = train_classification(train_loader2, val_loader2, net2, args2, logger)
torch.save(best_state_dict2,
           f'{param["save_model_path"]}{time_train}/model {mod2} param classification.pkl')

# load pretrained from dir
net1.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod1} param classification.pkl'))
net2.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod2} param classification.pkl'))

# train model by contrastive
training_set = ModalityDataset2(net1_input_train, net2_input_train, types='2mod')
val_set = ModalityDataset2(net1_input_val, net2_input_val, types='2mod')

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

best_state_dict1, best_state_dict2 = train_contrastive(train_loader, val_loader, net1, net2, args1, logger)

torch.save(best_state_dict1,
           f'{param["save_model_path"]}{time_train}/model {mod1} param contrastive.pkl')
torch.save(best_state_dict2,
           f'{param["save_model_path"]}{time_train}/model {mod2} param contrastive.pkl')

# load pretrained from dir
net1.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod1} param contrastive.pkl'))
net2.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod2} param contrastive.pkl'))

# train model to predict modality
training_set1 = ModalityDataset2(net1_input_train, net1_output_train, types='2mod')
training_set2 = ModalityDataset2(net2_input_train, net2_output_train, types='2mod')
val_set1 = ModalityDataset2(net1_input_val, net1_output_val, types='2mod')
val_set2 = ModalityDataset2(net2_input_val, net2_output_val, types='2mod')

train_loader1 = DataLoader(training_set1, **params)
train_loader2 = DataLoader(training_set2, **params)
val_loader1 = DataLoader(val_set1, **params)
val_loader2 = DataLoader(val_set2, **params)

best_state_dict1 = train_predict(train_loader1, val_loader1, net1, args1, logger, mod2_reducer)
torch.save(best_state_dict1,
           f'{param["save_model_path"]}{time_train}/model {mod1} param predict.pkl')

best_state_dict2 = train_predict(train_loader2, val_loader2, net2, args2, logger, mod1_reducer)
torch.save(best_state_dict2,
           f'{param["save_model_path"]}{time_train}/model {mod2} param predict.pkl')
