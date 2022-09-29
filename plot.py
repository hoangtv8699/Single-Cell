from utils import *

path = 'logs/28_09_2022 03_09_48.log'

classification_train_loss1, classification_val_loss1, classification_train_loss2, classification_val_loss2,\
    embed_train_loss, embed_test_loss, predict_train_loss1, predict_test_loss1, predict_train_loss2, predict_test_loss2 = read_logs(path)

plot_loss(classification_train_loss1, classification_val_loss1)
plot_loss(classification_train_loss2, classification_val_loss2)
plot_loss(embed_train_loss, embed_test_loss)
plot_loss(predict_train_loss1, predict_test_loss1)
plot_loss(predict_train_loss1, predict_test_loss1)

# import os
# import time
# import torch
# import pandas as pd
# import scanpy as sc
# import anndata as ad
# import logging
# import pickle as pk
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# from argparse import Namespace
# from torch.utils.data import DataLoader
#
# from datetime import datetime
#
#
# def embedding(mod, n_components, random_seed=0):
#     # sc.pp.log1p(mod)
#     # sc.pp.scale(mod)
#
#     mod_reducer = TruncatedSVD(n_components=n_components, random_state=random_seed)
#     truncated_mod = mod_reducer.fit_transform(mod)
#     del mod
#     return truncated_mod, mod_reducer
#
#
# dataset_path = 'data/train/cite/'
# pretrain_path = 'pretrain/'
#
# param = {
#     'use_pretrained': True,
#     'input_train_mod1': f'{dataset_path}adt.h5ad',
#     'input_train_mod2': f'{dataset_path}gex.h5ad',
#     'subset_pretrain1': f'{pretrain_path}GEX_subset.pkl',
#     'subset_pretrain2': f'{pretrain_path}ATAC_subset.pkl',
#     'output_pretrain': 'pretrain/',
#     'save_model_path': 'saved_model/',
#     'logs_path': 'logs/'
# }
#
# args1 = Namespace(
#     input_feats=32,
#     num_class=22,
#     embed_hid_feats=512,
#     latent_feats=32,
#     class_hid_feats=512,
#     pred_hid_feats=512,
#     out_feats=13431,
#     random_seed=17,
#     activation='relu',
#     act_out='sigmoid',
#     num_embed_layer=4,  # if using residual, this number must divisible by 2
#     num_class_layer=4,  # if using residual, this number must divisible by 2
#     num_pred_layer=4,  # if using residual, this number must divisible by 2
#     dropout=0.2,
#     epochs=1000,
#     lr_classification=1e-4,
#     lr_embed=1e-4,
#     lr_predict=1e-3,
#     normalization='batch',
#     patience=10
# )
#
# args2 = Namespace(
#     input_feats=256,
#     num_class=22,
#     embed_hid_feats=2048,
#     latent_feats=512,
#     class_hid_feats=2048,
#     pred_hid_feats=2048,
#     out_feats=13431,
#     random_seed=17,
#     activation='relu',
#     act_out='relu',
#     num_embed_layer=4,  # if using residual, this number must divisible by 2
#     num_class_layer=4,  # if using residual, this number must divisible by 2
#     num_pred_layer=4,  # if using residual, this number must divisible by 2
#     dropout=0.2,
#     epochs=1000,
#     lr_classification=1e-4,
#     lr_embed=1e-4,
#     lr_predict=1e-4,
#     normalization='batch',
#     patience=10
# )
#
# # get feature type
# train_mod1 = sc.read_h5ad(param['input_train_mod1'])
# train_mod2 = sc.read_h5ad(param['input_train_mod2'])
# mod1 = train_mod1.var['feature_types'][0]
# mod2 = train_mod2.var['feature_types'][0]
#
# # log norm train mod1
# # sc.pp.log1p(train_mod2)
#
# # get input and encode label for classification training
# LE = LabelEncoder()
# train_mod1.obs["class_label"] = LE.fit_transform(train_mod1.obs["cell_type"])
# input_label = train_mod1.obs["class_label"].to_numpy()
#
# mod1_train, _, mod2_train, _, label_train, label_val = train_test_split(train_mod1.X, train_mod2.X, input_label,
#                                                                         test_size=0.1,
#                                                                         random_state=args1.random_seed,
#                                                                         stratify=input_label)
#
# mod1_train, mod1_reducer = embedding(mod1_train, args1.input_feats, random_seed=args1.random_seed)
# pk.dump(mod1_reducer, open(f'pretrain/{mod1} reducer cite nolog.pkl', "wb"))
# mod2_train, mod2_reducer = embedding(mod2_train, args2.input_feats, random_seed=args2.random_seed)
# pk.dump(mod2_reducer, open(f'pretrain/{mod2} reducer cite nolog.pkl', "wb"))
