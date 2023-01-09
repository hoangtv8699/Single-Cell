import pickle as pk
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import sys
# from multithread import crawl
#
# np.set_printoptions(threshold=sys.maxsize)

path_atac = '../data/paper data/atac2gex/train_mod1.h5ad'
adata_atac = sc.read_h5ad(path_atac)
path_gex = '../data/paper data/atac2gex/train_mod2.h5ad'
adata_gex = sc.read_h5ad(path_gex)

# cell_type = 'CD8+ T'
#
# # print(adata_atac.obs.value_counts('cell_type'))
# adata_atac = adata_atac[adata_atac.obs['cell_type'] == cell_type, :]
# adata_gex = adata_gex[adata_gex.obs['cell_type'] == cell_type, :]
#
# for key in adata_atac.obs.keys():
#     if key not in ['cell_type', 'batch']:
#         del adata_atac.obs[key]
#         del adata_gex.obs[key]
#
# for key in adata_atac.var.keys():
#     if key not in ['cell_type', 'batch']:
#         del adata_atac.var[key]
#         del adata_gex.var[key]
#
# del adata_atac.uns
# del adata_atac.obsm
# del adata_gex.uns
# del adata_gex.obsm

gene_locus = pk.load(open('gene locus 2.pkl', 'rb'))
gene_dict = pk.load(open('gene infor 2.pkl', 'rb'))
# print(adata_atac)
# print(gene_locus)

list_gex = []
list_atac = []
for key in gene_locus.keys():
    if len(gene_locus[key]) > 0:
        # if int(gene_dict[key]['chromosome_name'][-3:]) == 1:
        list_gex.append(key)
        list_atac += gene_locus[key]

atacs = []
for atac in adata_atac.var_names:
    if atac in list_atac:
        atacs.append(atac)
gexs = []
for gex in adata_gex.var_names:
    if gex in list_gex:
        gexs.append(gex)

adata_atac = adata_atac[:, atacs]
adata_gex = adata_gex[:, gexs]

adata_atac.write(f'../data/paper data/atac2gex/atac_train.h5ad')
adata_gex.write(f'../data/paper data/atac2gex/gex_train.h5ad')


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