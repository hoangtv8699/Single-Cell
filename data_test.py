import random
import scanpy as sc
import pickle as pk

raw_path = 'data/multiome_BMMC_processed/raw.h5ad'
processed_path = 'data/train/multiome/gex.h5ad'

raw = sc.read_h5ad(raw_path)
processed = sc.read_h5ad(processed_path)

pretrain_path = 'pretrain/'

subset1 = pk.load(open(f'{pretrain_path}GEX_subset.pkl', 'rb'))
subset2 = pk.load(open(f'{pretrain_path}ATAC_subset.pkl', 'rb'))

input_reduced = processed[:, list(subset1)]
# print(input_train_mod2_reduced.shape)

print(processed.layers['counts'].shape)
print(raw.layers['counts'].shape)
print(input_reduced.X.shape)
#
# print(raw.var['feature_types'][0])
# print(processed.var['feature_types'][0])
#
# print(raw)
# print(processed)
#
# # train_mask = raw.obs_names[:processed.X.shape[0]]
# # test_mask = raw.obs_names[processed.X.shape[0]:]
#
# train_mask = []
# test_mask = []
#
# for x in raw.obs_names:
#     if x in processed.obs_names:
#         train_mask.append(x)
#     else:
#         test_mask.append(x)
#
# print(len(train_mask))
# print(len(test_mask))
#
# train = raw[train_mask]
# test = raw[test_mask]
#
# gex_train = train[:, train.var['feature_types'] == 'GEX']
# atac_train = train[:, train.var['feature_types'] == 'ADT']
# gex_test = test[:, test.var['feature_types'] == 'GEX']
# atac_test = test[:, test.var['feature_types'] == 'ADT']
#
# print(gex_train)
# print(atac_train)
# print(gex_test)
# print(atac_test)
#
# train_path = 'data/train/cite/'
# test_path = 'data/test/cite/'
#
# gex_train.write(train_path + 'gex.h5ad')
# atac_train.write(train_path + 'adt.h5ad')
# gex_test.write(test_path + 'gex.h5ad')
# atac_test.write(test_path + 'adt.h5ad')