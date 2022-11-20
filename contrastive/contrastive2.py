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

args = Namespace(
    num_class=22,
    latent_feats=16,
    pred_hid_feats=512,
    embed_hid_feats=512,
    random_seed=17,
    activation='relu',
    act_out='none',
    num_embed_layer=4,
    num_pred_layer=4,  # if using residual, this number must divisible by 2
    dropout=0.2,
    epochs=1000,
    lr_embed=1e-3,
    lr_predict=1e-4,
    normalization='batch',
    patience=10
)

# get feature type
train_mod1 = sc.read_h5ad(param['input_train_mod1'])
train_mod2 = sc.read_h5ad(param['input_train_mod2'])

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + f' {mod1} to {mod2}'
# time_train = '03_10_2022 22_30_28 gex to atac'
os.mkdir(f'{param["save_model_path"]}{time_train}')
logger = open(f'{param["logs_path"]}{time_train}.log', 'a')

mod1_reducer = pk.load(open(param['subset_pretrain1'], 'rb'))
mod2_reducer = pk.load(open(param['subset_pretrain2'], 'rb'))

# net1 input and output
net_input = csc_matrix(mod1_reducer.transform(train_mod1.X))
net_output = csc_matrix(mod2_reducer.transform(train_mod2.X))

# # # if not using reducer
# net1_input = train_mod1.X
# net1_output = train_mod2.X

args.input_feats = net_input.shape[1]
args.out_feats = net_output.shape[1]

logger.write('args1: ' + str(args) + '\n')
# dump args
pk.dump(args, open(f'{param["save_model_path"]}{time_train}/args net1.pkl', 'wb'))

net1_input_train, net1_input_val, net1_output_train, net1_output_val = train_test_split(net_input,
                                                                                        net_output,
                                                                                        test_size=0.1,
                                                                                        random_state=args.random_seed)

net1_input_train = sc.AnnData(net1_input_train, dtype=net1_input_train.dtype)
net1_input_val = sc.AnnData(net1_input_val, dtype=net1_input_val.dtype)
net1_output_train = sc.AnnData(net1_output_train, dtype=net1_output_train.dtype)
net1_output_val = sc.AnnData(net1_output_val, dtype=net1_output_val.dtype)

del train_mod1
del train_mod2

net = ContrastiveModel(args)
logger.write('net1: ' + str(net) + '\n')

params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 0}

# train model by contrastive
training_set = ModalityDataset2(net1_input_train, net1_output_train, types='2mod')
val_set = ModalityDataset2(net1_input_val, net1_output_val, types='2mod')

train_loader = DataLoader(training_set, **params)
val_loader = DataLoader(val_set, **params)

best_state_dict1 = train_contrastive2(train_loader, val_loader, net, args, logger)

torch.save(best_state_dict1,
           f'{param["save_model_path"]}{time_train}/model {mod1} param contrastive.pkl')

