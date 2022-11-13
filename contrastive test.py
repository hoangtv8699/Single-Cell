import pickle as pk

from scipy.sparse import csc_matrix
from torch.utils.data import DataLoader

from utils import *

# device = torch.device("cuda:0")

mod1 = 'gex'
mod2 = 'adt'


dataset_path = f'data/paper data/{mod1}2{mod2}/'
pretrain_path = f'pretrain/paper data/{mod1}2{mod2}/'

param = {
    'use_pretrained': True,
    'input_test_mod1': f'{dataset_path}test_mod1.h5ad',
    'input_test_mod2': f'{dataset_path}test_mod2.h5ad',
    'subset_pretrain1': f'{pretrain_path}mod1 reducer.pkl',
    'subset_pretrain2': f'{pretrain_path}mod2 reducer.pkl',
    'output_pretrain': 'pretrain/',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}

time_train = '30_10_2022 21_44_08 gex to adt'

# if load args
args1 = pk.load(open(f'{param["save_model_path"]}{time_train}/args net1.pkl', 'rb'))
args2 = pk.load(open(f'{param["save_model_path"]}{time_train}/args net2.pkl', 'rb'))

# get feature type
test_mod1 = sc.read_h5ad(param['input_test_mod1'])
test_mod2 = sc.read_h5ad(param['input_test_mod2'])

# select feature
mod1_reducer = pk.load(open(param['subset_pretrain1'], 'rb'))
mod2_reducer = pk.load(open(param['subset_pretrain2'], 'rb'))

params = {'batch_size': 2000,
          'shuffle': False,
          'num_workers': 0}

# test model mod 1
input = csc_matrix(mod1_reducer.transform(test_mod1.X))
# labels = csc_matrix(mod1_reducer.transform(test_mod2.X))

# input = test_mod1.X
labels = test_mod2.X

val_set = ModalityDataset(input, labels, types='2mod')
val_loader = DataLoader(val_set, **params)

net = ContrastiveModel(args1)
net.load_state_dict(torch.load(f'{param["save_model_path"]}{time_train}/model {mod1} param predict.pkl'))
net.cuda()
net.eval()

rmse = 0
pear = 0
with torch.no_grad():
    for val_batch, label in val_loader:
        val_batch, label = val_batch.cuda(), label.cuda()
        out = net(val_batch, residual=True, types='predict')
        out = mod2_reducer.inverse_transform(out.detach().cpu().numpy())
        rmse += mean_squared_error(label.detach().cpu().numpy(), out) * val_batch.size(0)

rmse = math.sqrt(rmse / len(val_loader.dataset))
print(rmse)