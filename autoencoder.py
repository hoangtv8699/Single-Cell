import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils import *

device = torch.device("cuda:0")

mod1 = 'gex'
mod2 = 'atac'
dataset_path = f'data/paper data/{mod1}2{mod2}/'
pretrain_path = f'pretrain/paper data/{mod1}2{mod2}/'

param = {
    'use_pretrained': True,
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'subset_pretrain1': f'{pretrain_path}mod1 reducer.pkl',
    'subset_pretrain2': f'{pretrain_path}mod2 reducer.pkl',
    'output_pretrain': 'pretrain/',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}

args1 = Namespace(
    num_class=22,
    latent_feats=64,
    pred_hid_feats=256,
    embed_hid_feats=256,
    random_seed=17,
    activation='relu',
    act_out='none',
    num_embed_layer=4,
    num_pred_layer=4,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=1000,
    lr_embed=1e-4,
    lr_predict=1e-4,
    normalization='batch',
    patience=10
)

args2 = Namespace(
    num_class=22,
    latent_feats=64,
    pred_hid_feats=256,
    embed_hid_feats=256,
    random_seed=17,
    activation='relu',
    act_out='none',
    num_embed_layer=4,
    num_pred_layer=4,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=1000,
    lr_embed=1e-4,
    lr_predict=1e-4,
    normalization='batch',
    patience=10
)

# get feature type
train_mod1 = sc.read_h5ad(param['input_train_mod1'])
train_mod2 = sc.read_h5ad(param['input_train_mod2'])

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + f' {mod1} to {mod2}'
# time_train = '27_09_2022 09_15_57 mod'
os.mkdir(f'{param["save_model_path"]}{time_train}')
logger = open(f'{param["logs_path"]}{time_train}.log', 'a')

mod1_reducer = pk.load(open(param['subset_pretrain1'], 'rb'))
mod2_reducer = pk.load(open(param['subset_pretrain2'], 'rb'))

# # net1 input and output
# net1_input = csc_matrix(mod1_reducer.transform(train_mod1.X))
# net1_output = train_mod2.X
# net2_input = csc_matrix(mod2_reducer.transform(train_mod2.X))
# net2_output = train_mod1.X

# # if not using reducer
net1_input = train_mod1.X
net1_output = train_mod2.X
net2_input = train_mod2.X
net2_output = train_mod1.X

args1.input_feats = net1_input.shape[1]
args1.out_feats = net1_input.shape[1]
args2.input_feats = net2_input.shape[1]
args2.out_feats = net2_input.shape[1]

logger.write('args1: ' + str(args1) + '\n')
logger.write('args2: ' + str(args2) + '\n')
# dump args
pk.dump(args1, open(f'{param["save_model_path"]}{time_train}/args net1.pkl', 'wb'))
pk.dump(args2, open(f'{param["save_model_path"]}{time_train}/args net2.pkl', 'wb'))

net1_input_train, net1_input_val, net1_output_train, net1_output_val, \
net2_input_train, net2_input_val, net2_output_train, net2_output_val = train_test_split(net1_input,
                                                                                        net1_output,
                                                                                        net2_input,
                                                                                        net2_output,
                                                                                        test_size=0.1,
                                                                                        random_state=args1.random_seed)

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

# train model by contrastive
training_set = ModalityDataset2(net1_input_train, net2_input_train, types='2mod')
val_set = ModalityDataset2(net1_input_val, net2_input_val, types='2mod')

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

best_state_dict1, best_state_dict2 = train_autoencoder(train_loader, val_loader, net1, net2, args1, logger)

torch.save(best_state_dict1,
           f'{param["save_model_path"]}{time_train}/model {mod1} param contrastive.pkl')
torch.save(best_state_dict2,
           f'{param["save_model_path"]}{time_train}/model {mod2} param contrastive.pkl')

