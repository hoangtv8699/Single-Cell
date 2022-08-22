import scanpy as sc

from sklearn.decomposition import TruncatedSVD


def embedding(mod, n_components, random_seed=0):
    sc.pp.log1p(mod)
    sc.pp.scale(mod)

    mod_reducer = TruncatedSVD(n_components=n_components, random_state=random_seed)
    mod_reducer.fit(mod)
    truncated_mod = mod_reducer.transform(mod)
    del mod
    return truncated_mod, mod_reducer

