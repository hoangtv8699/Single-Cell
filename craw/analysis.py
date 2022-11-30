import pickle as pk
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
from multithread import crawl

np.set_printoptions(threshold=sys.maxsize)

# path_atac = '../data/multiome/atac.h5ad'
# adata_atac = sc.read_h5ad(path_atac)
# path_gex = '../data/multiome/gex.h5ad'
# adata_gex = sc.read_h5ad(path_gex)
#
# cell_type = 'CD4+ T activated'
#
# # print(adata_atac.obs.value_counts('cell_type'))
# adata_atac = adata_atac[adata_atac.obs['cell_type'] == cell_type, :]
# adata_gex = adata_gex[adata_gex.obs['cell_type'] == cell_type, :]
#
# for key in adata_atac.obs.keys():
#     if key != 'cell_type':
#         del adata_atac.obs[key]
#         del adata_gex.obs[key]
#
# for key in adata_atac.var.keys():
#     del adata_atac.var[key]
#     del adata_gex.var[key]
#
#
# del adata_atac.uns
# del adata_atac.obsm
# del adata_gex.uns
# del adata_gex.obsm
#
# gene_locus = pk.load(open('gene locus new.pkl', 'rb'))
# gene_dict = pk.load(open('gene dict.pkl', 'rb'))
#
# gene_dict2 = {}
# for key in gene_locus.keys():
#     gene_dict2[key] = gene_dict[key]['position_in_chromosome'][0]
#
# print(adata_atac)
# print(gene_locus)
#
# # check crossing in gene
# for k1, v1 in sorted(gene_dict2.items(), key=lambda item: int(item[1]['chromosome_number']) * 1e9 + int(item[1]['chromosome_from'])):
#     for k2, v2 in sorted(gene_dict2.items(), key=lambda item: int(item[1]['chromosome_number']) * 1e9 + int(item[1]['chromosome_from'])):
#         if v1['chromosome_number'] == v2['chromosome_number']:
#             if int(v1['chromosome_from']) < int(v2['chromosome_from']) < int(v1['chromosome_to']):
#                 print(k1, k2)
#                 print(v1, v2)
#             elif int(v1['chromosome_from']) > int(v2['chromosome_to']):
#                 break
#
#
# list_atac = []
# for k, v in sorted(gene_dict2.items(), key=lambda item: int(item[1]['chromosome_number']) * 1e9 + int(item[1]['chromosome_from'])):
#     list_atac += gene_locus[k]
#
# set_atac = set(list_atac)
# print(len(list_atac))
# print(len(set_atac))
# adata_atac = adata_atac[:, list(set_atac)]
#
# adata_atac.write(f'../data/multiome/atac {cell_type}.h5ad')
# adata_gex.write(f'../data/multiome/gex {cell_type}.h5ad')


# # gene_atac = {}

# # for k in gene_locus.keys():
# #     gene_atac[k] = {}
# #     adata_tmp = adata_atac[:, gene_locus[k]]
# #     gene_atac[k]['binary'] = adata_tmp.X
# #     gene_atac[k]['raw'] = adata_tmp.layers['counts']

# # pk.dump(gene_atac, open('gene atac.pkl', 'wb'))


###### recreate train and test data with full metadata

# adata_atac_full = sc.read_h5ad('../data/multiome/atac.h5ad')
# # adata_gex_full = sc.read_h5ad('../data/multiome/gex.h5ad')
#
# adata_atac_paper = sc.read_h5ad('../data/paper data/atac2gex/test_mod1.h5ad')
# # adata_gex_paper = sc.read_h5ad('../data/paper data/atac2gex/test_mod2.h5ad')
#
# train_cell = adata_atac_full.obs_names[adata_atac_full.obs_names.isin(adata_atac_paper.obs_names)]
# adata_atac = adata_atac_full[train_cell]
# adata_atac = adata_atac[:, list(adata_atac_paper.var_names)]
#
# # train_cell = adata_gex_full.obs_names[adata_gex_full.obs_names.isin(adata_gex_paper.obs_names)]
# # adata_gex = adata_gex_full[train_cell]
# # adata_gex = adata_gex[:, list(adata_gex_paper.var_names)]
# #
# adata_atac.write('../data/multiome/atac2gex/test_mod1.h5ad')
# # adata_gex.write('../data/multiome/atac2gex/test_mod2.h5ad')

# adata_atac_paper = sc.read_h5ad('../data/multiome/atac2gex/train_mod1.h5ad')
# adata_gex_paper = sc.read_h5ad('../data/multiome/atac2gex/train_mod2.h5ad')
#
# print(adata_atac_paper)
# print(adata_gex_paper)