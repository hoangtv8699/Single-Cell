import random
import scanpy as sc
import pickle as pk
from scipy.sparse import csc_matrix
from sklearn.decomposition import TruncatedSVD


def embedding(mod, n_components, random_seed=0):
    mod_reducer = TruncatedSVD(n_components=n_components, random_state=random_seed)
    truncated_mod = mod_reducer.fit_transform(mod)
    del mod
    return truncated_mod, mod_reducer


path = '../data/paper data/'
folders = ['adt2gex/', 'gex2adt/', 'atac2gex/', 'gex2atac/']
path_pretrain = '../pretrain/paper data/'

for folder in folders:
    train_mod1 = sc.read_h5ad(f'{path+folder}train_mod1.h5ad')
    train_mod2 = sc.read_h5ad(f'{path+folder}train_mod2.h5ad')

    if folder == 'adt2gex/':
        mod1_train, mod1_reducer = embedding(train_mod1.X, 64, random_seed=17)
        pk.dump(mod1_reducer, open(f'{path_pretrain + folder}mod1 reducer.pkl', "wb"))
        mod2_train, mod2_reducer = embedding(train_mod2.X, 256, random_seed=17)
        pk.dump(mod2_reducer, open(f'{path_pretrain + folder}mod2 reducer.pkl', "wb"))
    elif folder == 'gex2adt/':
        mod1_train, mod1_reducer = embedding(train_mod1.X, 256, random_seed=17)
        pk.dump(mod1_reducer, open(f'{path_pretrain + folder}mod1 reducer.pkl', "wb"))
        mod2_train, mod2_reducer = embedding(train_mod2.X, 64, random_seed=17)
        pk.dump(mod2_reducer, open(f'{path_pretrain + folder}mod2 reducer.pkl', "wb"))
    else:
        mod1_train, mod1_reducer = embedding(train_mod1.X, 256, random_seed=17)
        pk.dump(mod1_reducer, open(f'{path_pretrain + folder}mod1 reducer.pkl', "wb"))
        mod2_train, mod2_reducer = embedding(train_mod2.X, 256, random_seed=17)
        pk.dump(mod2_reducer, open(f'{path_pretrain + folder}mod2 reducer.pkl', "wb"))