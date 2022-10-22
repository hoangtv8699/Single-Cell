import pickle as pk
from multiprocessing.dummy import Pool as ThreadPool

import scanpy as sc
import xmltodict
from Bio import Entrez

Entrez.email = 'nhoxkhang351@gmail.com'


def crawl(name):
    # search gene and get id
    handle = Entrez.esearch(db='GENE', term=f'{name}[Gene Name] AND "Homo sapiens"[Organism]')
    record = Entrez.read(handle)
    handle.close()

    if len(record['IdList']) == 0:
        # print(name + ' no data')
        return name, 'no data'
    # get relevant data
    try:
        fetch = Entrez.efetch(db='GENE', id=record['IdList'][0], retmode='xml')
        gb = fetch.read()
    except:
        # print(name + " bad request")
        return name, 'no data'

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
            # print(name + " no data in locus")
            return name, 'no data'

    res = {
        'position_in_chromosome': chr_in,
    }
    return name, res


path1 = '../data/paper data/gex2atac/train_mod1.h5ad'
path2 = '../data/paper data/gex2atac/train_mod2.h5ad'
train_mod1 = sc.read_h5ad(path1)
train_mod2 = sc.read_h5ad(path2)

pool = ThreadPool(100)
results = pool.map(crawl, train_mod1.var_names)

gene_dict = {}
for result in results:
    name, res = result
    gene_dict[name] = res

pk.dump(gene_dict, open('gene dict.pkl', 'wb'))
