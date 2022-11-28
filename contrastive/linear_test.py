import pickle as pk

from scipy.sparse import csc_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import math

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

time_train = '28_11_2022 16_47_12 atac to gex'
# time_train = '24_11_2022 13_31_02 atac to gex'
# args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

# get feature type
test_mod1_ori = sc.read_h5ad(param['input_test_mod1'])
test_mod2_ori = sc.read_h5ad(param['input_test_mod2'])

# if using raw data
test_mod1_ori.X = test_mod1_ori.layers['counts']

gene_locus = pk.load(open('../craw/gene locus 2.pkl', 'rb'))
# gene_list = list(gene_locus.keys())
gene_list = os.listdir(f'{param["save_model_path"]}{time_train}')
# gene_name = gene_list[0]

excellent = []
good = []
bad = []

for gene_name in gene_list:
    gene_name = gene_name.split('.')[0]
    if gene_name == 'args net':
        continue

    test_mod1 = test_mod1_ori[:, gene_locus[gene_name]]
    test_mod2 = test_mod2_ori[:, gene_name]

    params = {'batch_size': 256,
              'shuffle': False,
              'num_workers': 0}

    # normalize data
    sc.pp.log1p(test_mod1)
    sc.pp.scale(test_mod1)

    mod1 = test_mod1.X
    mod2 = test_mod2.X.toarray()

    # test sklearn LR model
    net = pk.load(open(f'{param["save_model_path"]}{time_train}/{gene_name}.pkl', 'rb'))
    out = net.predict(mod1)

    # rmse = mean_squared_error(mod2, out, squared=False)
    rmse = cal_rmse(csc_matrix(mod2), csc_matrix(out))
    print(rmse)

    if rmse < 0.1:
        excellent.append(gene_name)
    elif rmse < 0.2:
        good.append(gene_name)
    else:
        bad.append(gene_name)
    # break

    # # test torch model
    # mod1 = torch.tensor(mod1).float().cuda()
    # mod2 = torch.tensor(mod2).float()
    #
    # net = LinearRegressionModel(mod1.shape[1], mod2.shape[1])
    # net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/{gene_name}.pkl'))
    # net.cuda()
    # net.eval()
    #
    # out = net(mod1)
    # # rmse = mean_squared_error(mod2.numpy(), out.detach().cpu().numpy(), squared=False)
    # rmse = cal_rmse(csc_matrix(mod2.numpy()), csc_matrix(out.detach().cpu().numpy()))
    # print(rmse)


print(len(excellent), " < 0.1")
print(len(good), "0.1 < < 0.2")
print(len(bad), " > 0.2")

pk.dump(excellent, open('excellent.pkl', 'wb'))
pk.dump(good, open('good.pkl', 'wb'))
pk.dump(bad, open('bad.pkl', 'wb'))
