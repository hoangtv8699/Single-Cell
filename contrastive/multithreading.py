from argparse import Namespace
from datetime import datetime

from utils import *

device = torch.device("cuda:0")


num_thread = 10
mod1_name = 'atac'
mod2_name = 'gex'
dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'

param = {
    'use_pretrained': True,
    'input_train_mod1': f'{dataset_path}train_mod1.h5ad',
    'input_train_mod2': f'{dataset_path}train_mod2.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}

args = Namespace(
    random_seed=17,
    epochs=1,
    lr=1e-3,
    patience=10
)

now = datetime.now()
time_train = now.strftime("%d_%m_%Y %H_%M_%S") + f' {mod1_name} to {mod2_name}'
# time_train = '03_10_2022 22_30_28 gex to atac'
os.mkdir(f'{param["save_model_path"]}{time_train}')
os.mkdir(f'{param["logs_path"]}{time_train}')

train_mod1_ori = sc.read_h5ad(param['input_train_mod1'])
train_mod2_ori = sc.read_h5ad(param['input_train_mod2'])

# # get filter cell type
# cell_type = 'CD4+ T activated'
# train_mod1_ori = train_mod1_ori[train_mod1_ori.obs['cell_type'] == cell_type, :]
# train_mod2_ori = train_mod2_ori[train_mod2_ori.obs['cell_type'] == cell_type, :]

gene_locus = pk.load(open('../data/gene locus new.pkl', 'rb'))
gene_list = list(gene_locus.keys())

arr = np.arange(len(gene_list))
splitted_arr = np.array_split(arr, num_thread)

threads = []
for i in range(num_thread):
    gene_list_new = gene_list[splitted_arr[i][0]:splitted_arr[i][-1]]
    atac_list = []
    for gene in gene_list_new:
        atac_list += gene_locus[gene]
    atac_list = set(atac_list)
    threads.append(threading.Thread(target=train, name='thread ' + str(i),
                                    args=(time_train, param, args, gene_list_new, gene_locus,
                                          train_mod1_ori[:, list(atac_list)],
                                          train_mod2_ori[:, gene_list_new])))

del train_mod1_ori
del train_mod2_ori

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()
print("Done!")
