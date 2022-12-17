import pickle as pk

from scipy.sparse import csc_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error


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
    'subset_pretrain1': f'{pretrain_path}mod1 reducer.pkl',
    'subset_pretrain2': f'{pretrain_path}mod2 reducer.pkl',
    'output_pretrain': '../pretrain/',
    'save_model_path': '../saved_model/',
    'logs_path': '../logs/'
}

time_train = '26_11_2022 19_05_59 atac to gex'

# if load args
args = pk.load(open(f'{param["save_model_path"]}{time_train}/args net.pkl', 'rb'))

# get feature type
test_mod1 = sc.read_h5ad(param['input_test_mod1'])
test_mod2 = sc.read_h5ad(param['input_test_mod2'])

# select feature
mod1_reducer = pk.load(open(param['subset_pretrain1'], 'rb'))
mod2_reducer = pk.load(open(param['subset_pretrain2'], 'rb'))

params = {'batch_size': 256,
          'shuffle': False,
          'num_workers': 0}

# test model mod 1
# mod1 = csc_matrix(mod1_reducer.transform(test_mod1.X))
# mod2  = test_mod2.X
# mod1 = test_mod1.X
# mod2 = csc_matrix(mod2_reducer.transform(test_mod2.X))

mod1 = test_mod1.X.toarray()
mod2 = test_mod2.X.toarray()

mod1 = torch.tensor(mod1).float().cuda()
mod2 = torch.tensor(mod2).float()

net = ContrastiveModel2(args)
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model param predict.pkl'))
net.cuda()
net.eval()

out = net(mod1, types='1to2')
rmse = mean_squared_error(mod2.numpy(), out.detach().cpu().numpy(), squared=False)
rmse = cal_rmse(csc_matrix(mod2.numpy()), csc_matrix(out.detach().cpu().numpy()))
print(rmse)

# mod1 = sc.AnnData(mod1, dtype=mod1.dtype)
# mod2 = sc.AnnData(mod2, dtype=mod2.dtype)
#
# val_set = ModalityDataset2(mod1, mod2)
# val_loader = DataLoader(val_set, **params)

# net = ContrastiveModel2(args)
# net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model param predict.pkl'))
# net.cuda()
# net.eval()
#
# rmse1 = 0
# rmse2 = 0
# pear = 0
# with torch.no_grad():
#     for mod1, mod2 in val_loader:
#         mod1, mod2 = mod1.cuda(), mod2.cuda()
#         # out1 = net(mod1, types='1to2')
#         # out2 = net(mod2, types='2to1')
#         # rmse1 += (mean_squared_error(mod2.detach().cpu().numpy(), out1.detach().cpu().numpy()) * mod1.size(0))
#         # rmse2 += (mean_squared_error(mod1.detach().cpu().numpy(), out2.detach().cpu().numpy()) * mod1.size(0))
#
#         # out1 = net(mod2, types='2to1')
#         # out1 = mod1_reducer.inverse_transform(out1.detach().cpu().numpy())
#         # rmse1 += (mean_squared_error(mod1.detach().cpu().numpy(), out1) * mod1.size(0))
#
# rmse1 = math.sqrt(rmse1 / len(val_loader.dataset))
# rmse2 = math.sqrt(rmse2 / len(val_loader.dataset))
# print(rmse1)
# print(rmse2)
