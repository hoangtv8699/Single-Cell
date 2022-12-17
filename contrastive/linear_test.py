import pickle as pk

from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize

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

time_train = '03_12_2022 18_14_22 atac to gex'
# args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

# get feature type
test_mod1_ori = sc.read_h5ad(param['input_test_mod1'])
test_mod2_ori = sc.read_h5ad(param['input_test_mod2'])

# # if using raw data
# test_mod1_ori.X = normalize(test_mod1_ori.layers['counts'], axis=0)

gene_locus = pk.load(open('../craw/gene locus promoter.pkl', 'rb'))
# gene_list = list(gene_locus.keys())
gene_list = os.listdir(f'{param["save_model_path"]}{time_train}')
# gene_name = gene_list[0]

excellent = []
good = []
bad = []

pred_matrix = []
gene_names = []

# gene_list = ['AAK1']
# gene_locus['AAK1'] = ['chr2-69582036-69582843', 'chr2-69591589-69592502']

i = 0
for gene_name in gene_list:
    gene_name = gene_name.split('.')[0]
    if gene_name == 'args net':
        continue
    gene_names.append(gene_name)

    test_mod1 = test_mod1_ori[:, gene_locus[gene_name]]
    test_mod2 = test_mod2_ori[:, gene_name]

    params = {'batch_size': 256,
              'shuffle': False,
              'num_workers': 0}

    mod1 = test_mod1.X.toarray()
    mod2 = test_mod2.X.toarray()

    # test sklearn LR model
    net = pk.load(open(f'{param["save_model_path"]}{time_train}/{gene_name}.pkl', 'rb'))
    out = net.predict(mod1)
    # if len(pred_matrix) == 0:
    #     pred_matrix = out
    # else:
    #     pred_matrix = np.append(pred_matrix, out, axis=1)
    # i += 1
    # print(i)

    # # rmse = mean_squared_error(mod2, out, squared=False)
    rmse = cal_rmse(csr_matrix(mod2), csr_matrix(out))
    # print(rmse)
    #
    # break
    #
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
#
# pk.dump(excellent, open('excellent.pkl', 'wb'))
# pk.dump(good, open('good.pkl', 'wb'))
# pk.dump(bad, open('bad.pkl', 'wb'))

# mod2_processed = test_mod2_ori[:, gene_names]
# mod2_processed.write('../data/processed/atac2gex/train_mod2.h5ad')
#
# mod1_processed = sc.AnnData(
#     X=csr_matrix(pred_matrix),
#     obs=mod2_processed.obs,
#     var=mod2_processed.var,
# )
# mod1_processed.write('../data/processed/atac2gex/train_mod1.h5ad')


