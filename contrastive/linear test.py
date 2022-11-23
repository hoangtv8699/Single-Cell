import pickle as pk

from scipy.sparse import csc_matrix
from torch.utils.data import DataLoader

from utils import *

# device = torch.device("cuda:0")

mod1_name = 'atac'
mod2_name = 'gex'

dataset_path = f'../data/paper data/{mod1_name}2{mod2_name}/'
pretrain_path = f'../pretrain/paper data/{mod1_name}2{mod2_name}/'

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}

time_train = '23_11_2022 15_40_55 atac to gex'
args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

# get feature type
test_mod1_ori = sc.read_h5ad(param['input_test_mod1'])
test_mod2_ori = sc.read_h5ad(param['input_test_mod2'])

gene_locus = pk.load(open('../data/gene locus new.pkl', 'rb'))
gene_list = list(gene_locus.keys())
# gene_name = gene_list[0]

for gene_name in gene_list:
    test_mod1 = test_mod1_ori[:, gene_locus[gene_name]]
    test_mod2 = test_mod2_ori[:, gene_name]

    params = {'batch_size': 256,
              'shuffle': False,
              'num_workers': 0}

    mod1 = test_mod1.X
    mod2 = test_mod2.X

    mod1 = sc.AnnData(mod1, dtype=mod1.dtype)
    mod2 = sc.AnnData(mod2, dtype=mod2.dtype)

    val_set = ModalityDataset2(mod1, mod2)
    val_loader = DataLoader(val_set, **params)

    net = LinearRegressionModel(mod1.shape[1], mod2.shape[1])
    net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/{gene_name}.pkl'))
    net.cuda()
    net.eval()

    rmse = 0
    pear = 0
    with torch.no_grad():
        for mod1, mod2 in val_loader:
            mod1, mod2 = mod1.cuda(), mod2.cuda()
            out = net(mod1)
            rmse += (mean_squared_error(mod2.detach().cpu().numpy(), out.detach().cpu().numpy()) * mod1.size(0))

    rmse = math.sqrt(rmse / len(val_loader.dataset))
    print(f'{gene_name}: {rmse}')
