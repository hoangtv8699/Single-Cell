from selenium import webdriver
from bs4 import BeautifulSoup
import pickle as pk

gene_dict = pk.load(open('../craw/gene infor 2.pkl', 'rb'))
driver = webdriver.Chrome()

for gene_name in gene_dict:
    if gene_dict[gene_name] == 'not found' or gene_dict[gene_name] == 'no data in locus':
        continue

    name = gene_dict[gene_name]['chromosome_name']
    version = gene_dict[gene_name]['chromosome_assembly_version']
    start = int(gene_dict[gene_name]['chromosome_from']) + 1
    stop = int(gene_dict[gene_name]['chromosome_to']) + 1
    strand = gene_dict[gene_name]['strand']

    link = f'https://www.ncbi.nlm.nih.gov/nuccore/{name}.{version}?report=genbank&from={start}&to={stop}&strand=true'

    print(gene_dict[gene_name])
    # driver.get("https://www.ncbi.nlm.nih.gov/nuccore/NC_000002.12?report=genbank&from=69457997&to=69643739&strand=true")
    #
    # content = driver.page_source
    # soup = BeautifulSoup(content)
    # cds = soup.find('span', attrs={'id': 'feature_NC_000002.12_CDS_0'})
