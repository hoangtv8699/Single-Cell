import logging
import os
import pickle as pk
from argparse import Namespace
from datetime import datetime

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from utils import *


def checkpromoter(gene_name, gene_locus, gene_dict):
    start = int(gene_dict[gene_name]['chromosome_from'])
    stop = int(gene_dict[gene_name]['chromosome_to'])
    strand = gene_dict[gene_name]['strand']

    locus_arr = gene_locus[gene_name]

    for locus in locus_arr:
        tmp = locus.split('-')
        if strand == 'plus':
            stop = start
            start = start - 1500
        else:
            start = stop
            stop = stop + 1500
        if (int(tmp[1]) > start and int(tmp[2]) < stop) or (int(tmp[1]) < start < int(tmp[2]) < stop) or (
                start < int(tmp[1]) < stop < int(tmp[2])):
            return True
    return False


adata_atac = sc.read_h5ad('../data/paper data/atac2gex/train_mod1.h5ad')
# adata_gex = sc.read_h5ad('../data/paper data/atac2gex/train_mod2.h5ad')
gene_locus = pk.load(open('../craw/gene locus promoter.pkl', 'rb'))
gene_dict = pk.load(open('../craw/gene infor 2.pkl', 'rb'))
excellent = pk.load(open('excellent.pkl', 'rb'))
good = pk.load(open('good.pkl', 'rb'))
bad = pk.load(open('bad.pkl', 'rb'))

# print(gene_locus)
gene_list = []
for gene_name in gene_locus:
    if 0 < len(gene_locus[gene_name]):
        gene_list.append(gene_name)
print(len(gene_list))

# gene_name = 'AAK1'
# test_mod1 = adata_atac[:, gene_locus[gene_name]]
# test_mod2 = adata_gex[:, gene_name]
#
# print(test_mod2.layers['counts'].max())

# for i in range(test_mod1.X.shape[0]):
#     print(test_mod1[i].layers['counts'].toarray(), test_mod2[i].layers['counts'].toarray())

# print(bad)

# cds = [[700, 862], [86762, 86880], [99196, 99304], [101075, 101217],
#        [111578, 111699], [113034, 113115], [113600, 113732],
#        [116421, 116524], [118628, 118707], [122752, 122906],
#        [124500, 124786], [128991, 129269], [134280, 134483],
#        [182029, 182073]]
#
# print(gene_dict['AAK1'])
# print(gene_locus['AAK1'])
#
# start_ori = int(gene_dict['AAK1']['chromosome_from'])
# stop_ori = int(gene_dict['AAK1']['chromosome_to'])
#
# for locus in gene_locus['AAK1']:
#     tmp = locus.split('-')
#     for cd in cds:
#         start = start_ori + cd[0]
#         stop = start_ori + cd[1]
#         if (start < int(tmp[1]) and int(tmp[2]) < stop) or \
#                 (int(tmp[1]) < start < int(tmp[2]) < stop) or \
#                 (start < int(tmp[1]) < stop < int(tmp[2])) or \
#                 (int(tmp[1]) < start < stop < int(tmp[2])):
#             print(locus, cd)
        # if int(tmp[1]) < stop:
        #     break
# is_promoter = []
# is_not_promoter = []
#
# for gene_name in bad:
#     if checkpromoter(gene_name, gene_locus, gene_dict):
#         is_promoter.append(gene_name)
#     else:
#         is_not_promoter.append(gene_name)
#
# print(len(is_promoter), ' is promoter')
# print(len(is_not_promoter), ' is not promoter')

# gene_name = 'AAK1'
# print(checkpromoter(gene_name, gene_locus, gene_dict))
# print(gene_name)
# print(gene_locus[gene_name])
# print(gene_dict[gene_name]['chromosome_from'])
# print(gene_dict[gene_name]['chromosome_to'])
# print(gene_dict[gene_name]['strand'])
#
# max_rmse = 0
# min_rmse = 1
# arr_rmse = []
#
# mod1_name = 'atac'
# mod2_name = 'gex'
#
# dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
# pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'
#
# param = {
#     'use_pretrained': True,
#     'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
#     'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
#     'output_pretrain': '../pretrain/',
#     'save_model_path': '../saved_model/',
#     'logs_path': '../logs/'
# }
#
# test_mod1_ori = sc.read_h5ad(param['input_test_mod1'])
# test_mod2_ori = sc.read_h5ad(param['input_test_mod2'])
#
# # # if using raw data
# # test_mod1_ori.X = normalize(test_mod1_ori.layers['counts'], axis=0)
#
# for gene_name in gene_list:
#
#     # cell_type = 'CD4+ T activated'
#     # test_mod1_ori = test_mod1_ori[test_mod1_ori.obs['cell_type'] == cell_type, :]
#     # test_mod2_ori = test_mod2_ori[test_mod2_ori.obs['cell_type'] == cell_type, :]
#
#     test_mod1 = test_mod1_ori[:, gene_locus[gene_name]]
#     test_mod2 = test_mod2_ori[:, gene_name]
#
#     mod1 = test_mod1.X.toarray()
#     mod2 = test_mod2.X.toarray()
#
#     # test sklearn LR model
#     net = pk.load(open(f'../saved_model/03_12_2022 18_14_22 atac to gex/{gene_name}.pkl', 'rb'))
#     out = net.predict(mod1)
#
#     # rmse = mean_squared_error(mod2, out, squared=False)
#     rmse = cal_rmse(csc_matrix(mod2), csc_matrix(out))
#     # print(rmse)
#     arr_rmse.append(rmse)
#     max_rmse = max(max_rmse, rmse)
#     min_rmse = min(min_rmse, rmse)
#
# print(max_rmse)
# print(min_rmse)
#
#
# plt.hist(arr_rmse, bins=100)
# plt.show()

# # print histogram of length
# max_length = 0
# min_length = 113
# arr_length = []
# for gene_name in bad:
#     arr_length.append(len(gene_locus[gene_name]))
#     max_length = max(max_length, len(gene_locus[gene_name]))
#     min_length = min(min_length, len(gene_locus[gene_name]))
# print(max_length)
# print(min_length)
# plt.hist(arr_length, bins=100)
# plt.show()


# # create pearson correlation matrix
# mod1_processed = sc.read_h5ad('../data/processed/atac2gex/train_mod1.h5ad')
#
# data = mod1_processed.X.toarray()
# print(data.shape)
# corr = np.corrcoef(data.T)
# print(corr.shape)
#
# pk.dump(corr, open('../data/corrcoef_matrix.pkl', 'wb'))

# # # create edge and edge weight from coef matrix
# corr = pk.load(open('../data/corrcoef_matrix.pkl', 'rb'))
#
# src = []
# dst = []
# weights = []
#
# for i in range(corr.shape[0]):
#     for j in range(i, corr.shape[1]):
#         if abs(corr[i, j]) > 0.3:
#             src.append(i)
#             dst.append(j)
#             weights.append([corr[i, j]])
# pk.dump([src, dst, weights], open('../data/graph.pkl', 'wb'))
