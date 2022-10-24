import pickle as pk
import scanpy as sc


# path = '../data/multiome_BMMC_processed.h5ad'
# adata = sc.read_h5ad(path)
#
# print(adata)
#
# adata_gex = adata[:, adata.var['feature_types'] == 'GEX']
# adata_atac = adata[:, adata.var['feature_types'] == 'ATAC']
#
# adata_gex.write('../data/multiome/gex.h5ad')
# adata_atac.write('../data/multiome/atac.h5ad')


path_atac = '../data/multiome/atac.h5ad'
path_gex = '../data/multiome/gex.h5ad'
adata_atac = sc.read_h5ad(path_atac)
adata_gex = sc.read_h5ad(path_gex)

chr = {}
for var_name in adata_atac.var_names:
    k = var_name.split('-')[0]
    if k in chr.keys():
        chr[k].append(var_name)
    else:
        chr[k] = [var_name]

gene_dict = pk.load(open('gene dict 2.pkl', 'rb'))
gene_locus = {}

del_key = []

for key in gene_dict.keys():
    if gene_dict[key] == 'no data' or len(gene_dict[key]['position_in_chromosome']) == 0:
        del_key.append(key)
        continue
    start = gene_dict[key]['position_in_chromosome'][0]['chromosome_from']
    stop = gene_dict[key]['position_in_chromosome'][0]['chromosome_to']
    chr_key = 'chr' + gene_dict[key]['position_in_chromosome'][0]['chromosome_number']

    gene_locus[key] = []

    locus_arr = chr[chr_key]
    for locus in locus_arr:
        tmp = locus.split('-')
        # save atac if atac in the locus of gene
        if int(stop) > int(tmp[1]) > int(start) or int(stop) > int(tmp[2]) > int(start):
            gene_locus[key].append(locus)
        # stop search because it passed the locus
        if int(tmp[1]) > int(stop):
            break

for key in del_key:
    del gene_dict[key]

pk.dump(gene_locus, open('gene locus.pkl', 'wb'))

input = []
output = []
output_key = []
for k, v in sorted(gene_dict.items(), key=lambda item: item[1]['position_in_chromosome'][0]['chromosome_number'] * 1e9 + item[1]['position_in_chromosome'][0]['chromosome_from']):
    list_atac = gene_locus[k]
    input_tmp = adata_atac[list_atac]
    input.append(input_tmp.X.toarray())
    output_key.append(k)

output = adata_gex[output_key].X


