from Bio import Entrez
from Bio import SeqIO
import json
import xmltodict
# import asn1vnparser
import scanpy as sc
import pickle as pk

Entrez.email = 'nhoxkhang351@gmail.com'


def crawl(name):
    # search gene and get id
    handle = Entrez.esearch(db='GENE', term=f'{name}[Gene Name] AND "Homo sapiens"[Organism]')
    record = Entrez.read(handle)
    handle.close()

    if len(record['IdList']) == 0:
        print(name + ' no data')
        return 'no data'
    # get relevant data
    try:
        fetch = Entrez.efetch(db='GENE', id=record['IdList'][0], retmode='xml')
        gb = fetch.read()
    except:
        print(name + " bad request")
        return 'no data'

    # parse to dict for easy extract
    y = xmltodict.parse(gb)

    # get list potision of gene in chromosome
    chr_in = {}
    chr_list = y['Entrezgene-Set']['Entrezgene']['Entrezgene_locus']['Gene-commentary']

    if not isinstance(chr_list, list):
        chr_list = [chr_list]

    for chro in chr_list:
        try:
            version = chro['Gene-commentary_heading']
            if version != 'Reference GRCh38.p14 Primary Assembly':
                continue
            chr_number = chro['Gene-commentary_type']['#text']
            chr_name = chro['Gene-commentary_accession']
            chr_assem_ver = chro['Gene-commentary_version']
            chr_from = chro['Gene-commentary_seqs']['Seq-loc']['Seq-loc_int']['Seq-interval']['Seq-interval_from']
            chr_to = chro['Gene-commentary_seqs']['Seq-loc']['Seq-loc_int']['Seq-interval']['Seq-interval_to']
            strand = chro['Gene-commentary_seqs']['Seq-loc']['Seq-loc_int']['Seq-interval']['Seq-interval_strand'][
                'Na-strand']['@value']

            chr_in = {
                'version': version,
                'strand': strand,
                'chromosome_number': chr_number,
                'chromosome_name': chr_name,
                'chromosome_assembly_version': chr_assem_ver,
                'chromosome_from': chr_from,
                'chromosome_to': chr_to
            }
        except Exception as e:
            continue

    if not chr_in:
        print(name + " no data in locus")
        return 'no data'

    return chr_in


# ###### crawl
# path = '../data/paper data/gex2atac/train_mod1.h5ad'
# train_mod = sc.read_h5ad(path)
#
# gene_dict = {}
#
# for name in train_mod.var_names:
#     try:
#         res = crawl(name)
#         gene_dict[name] = res
#     except:
#         gene_dict[name] = 'no data'
#         print(name)
#
# pk.dump(gene_dict, open('gene infor.pkl', 'wb'))
# print(f'total: {len(list(gene_dict.keys()))}')

# analysis
gene_dict = pk.load(open('gene infor 2.pkl', 'rb'))

path1 = '../data/paper data/atac2gex/train_mod2.h5ad'
path2 = '../data/paper data/atac2gex/train_mod1.h5ad'
adata_gex = sc.read_h5ad(path1)
adata_atac = sc.read_h5ad(path2)

arr = []
for key in gene_dict.keys():
    if gene_dict[key] == 'not found' or gene_dict[key] == 'no data in locus':
        continue
    arr.append(gene_dict[key]['chromosome_name'])

arr = set(arr)
print(arr)
locus_arr = adata_atac.var_names
gene_locus = {}

count = 0

for key in gene_dict.keys():
    if gene_dict[key] == 'not found' or gene_dict[key] == 'no data in locus':
        continue
    start = int(gene_dict[key]['chromosome_from'])
    stop = int(gene_dict[key]['chromosome_to'])
    chr_number = int(gene_dict[key]['chromosome_name'][-6:])
    strand = gene_dict[key]['strand']

    gene_locus[key] = []

    for locus in locus_arr:
        tmp = locus.split('-')
        if tmp[0] == 'chr' + str(chr_number):
            # save atac if atac in the locus of gene's promoter, only 1500bp from 5'
            if strand == 'plus':
                stop = start
                start = start - 1500
            else:
                start = stop
                stop = stop + 1500
            # save atac if atac in the locus of gene
            if (start < int(tmp[1]) and int(tmp[2]) < stop) or \
                    (int(tmp[1]) < start < int(tmp[2]) < stop) or \
                    (start < int(tmp[1]) < stop < int(tmp[2])) or \
                    (start < int(tmp[1]) < int(tmp[2]) < stop):
                gene_locus[key].append(locus)
                count += 1
            # stop search because it passed the locus
            if int(tmp[1]) > stop:
                break

print(len(gene_locus.keys()))
print(count)
pk.dump(gene_locus, open('gene locus promoter.pkl', 'wb'))
