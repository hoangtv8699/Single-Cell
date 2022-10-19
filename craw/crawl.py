from Bio import Entrez
from Bio import SeqIO
import json
import xmltodict
import asn1vnparser
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
    chr_in = []
    chr_list = y['Entrezgene-Set']['Entrezgene']['Entrezgene_locus']['Gene-commentary']

    if not isinstance(chr_list, list):
        chr_list = [chr_list]

    for chro in chr_list:
        try:
            chr_number = chro['Gene-commentary_type']['#text']
            chr_name = chro['Gene-commentary_accession']
            chr_assem_ver = chro['Gene-commentary_version']
            chr_from = chro['Gene-commentary_seqs']['Seq-loc']['Seq-loc_int']['Seq-interval']['Seq-interval_from']
            chr_to = chro['Gene-commentary_seqs']['Seq-loc']['Seq-loc_int']['Seq-interval']['Seq-interval_to']

            chr_dict = {
                'chromosome_number': chr_number,
                'chromosome_name': chr_name,
                'chromosome_assembly_version': chr_assem_ver,
                'chromosome_from': chr_from,
                'chromosome_to': chr_to
            }
            chr_in.append(chr_dict)
        except:
            print(name + " no data in locus")

    res = {
        'position_in_chromosome': chr_in,
    }
    return res


####### crawl

# path = '../data/paper data/gex2adt/train_mod1.h5ad'
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
# pk.dump(gene_dict, open('gene dict.pkl', 'wb'))


# ###### crawl unexpected part
# with open('error.txt', 'rb') as f:
#     gene_dict = {}
#     lines = f.readlines()
#     for line in lines:
#         line = line.rstrip()
#         try:
#             res = crawl(line.decode("utf-8"))
#             gene_dict[line] = res
#         except:
#             gene_dict[line] = 'no data'
#             print(line.decode("utf-8") + " special")
#     pk.dump(gene_dict, open('gene dict 2.pkl', 'wb'))

###### crawl unexpected part
# with open('error 2.txt', 'rb') as f:
#     gene_dict = {}
#     lines = f.readlines()
#     names = []
#     for line in lines:
#         line = line.rstrip().decode("utf-8")
#         line = line.split(" ")
#         if line[1] == 'special':
#             names.append(line[0])
#
#     for name in names:
#         res = crawl(name)
#         gene_dict[name] = res
#     pk.dump(gene_dict, open('gene dict 3.pkl', 'wb'))

##### ensemble part
# gene = pk.load(open('gene dict.pkl', 'rb'))
# gene2 = pk.load(open('gene dict 2.pkl', 'rb'))
# gene3 = pk.load(open('gene dict 3.pkl', 'rb'))
#
# for key in gene2.keys():
#     gene[key] = gene2[key]
#
# for key in gene3.keys():
#     gene[key] = gene3[key]
#
# pk.dump(gene, open('gene dict.pkl', 'wb'))

# analysis
gene = pk.load(open('gene dict.pkl', 'rb'))

path1 = '../data/paper data/gex2atac/train_mod1.h5ad'
path2 = '../data/paper data/gex2atac/train_mod2.h5ad'
train_mod1 = sc.read_h5ad(path1)
train_mod2 = sc.read_h5ad(path2)

print(train_mod2.var_names[5])
print(gene[train_mod1.var_names[105]])
chr_number = 'chr' + gene[train_mod1.var_names[105]]['position_in_chromosome'][0]['chromosome_number']
chr_name = gene[train_mod1.var_names[105]]['position_in_chromosome'][0]['chromosome_name']
start_in = gene[train_mod1.var_names[105]]['position_in_chromosome'][0]['chromosome_from']
stop_in = gene[train_mod1.var_names[105]]['position_in_chromosome'][0]['chromosome_to']

# for var_name in train_mod2.var_names:
#     var_name = var_name.split('-')
#
#     if chr_number == var_name[0] or chr_name == var_name[0]:
#         if int(var_name[1]) > int(start_in) and int(var_name[2]) < int(stop_in):
#             print(var_name)

for var_name in train_mod2.var_names:
    print(var_name)