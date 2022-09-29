import time
import dgl
import torch
import pandas as pd
import scanpy as sc
import anndata as ad
import logging
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from argparse import Namespace

from datetime import datetime

from utils import *

dataset_path = '../data/cite_BMMC_processed/'
pretrain_path = 'pretrain/cite_BMMC_processed/'

param = {
    'use_pretrained_data': False,
    'input_train_mod1': f'{dataset_path}mod1.h5ad',
    'input_train_mod2': f'{dataset_path}mod2.h5ad',
    'output_pretrain': 'pretrain',
    'input_train_mod1_pretrained': f'{pretrain_path}mod1_train_svd.pkl',
    'input_train_mod2_pretrained': f'{pretrain_path}mod2_train_svd.pkl',
    'input_test_mod1_pretrained': f'{pretrain_path}mod1_test_svd.pkl',
    'input_test_mod2_pretrained': f'{pretrain_path}mod2_test_svd.pkl',
    'mod1_reducer_pretrained': f'{pretrain_path}svd_mod1.pkl',
    'mod2_reducer_pretrained': f'{pretrain_path}svd_mod2.pkl',
    'save_model_path': 'saved_model/',
    'logs_path': 'logs/'
}

args = Namespace(
    mod1_feat_size=100,
    mod2_feat_size=100,
    out_mod1_feats=100,
    out_mod2_feats=100,
    random_seed=17,
    hid_feats=100,
    latent_feats=20,
    num_gcn1_layer=7,  # if using residual, this number must be odd
    num_gcn2_layer=7,  # if using residual, this number must be odd
    activation='relu',
    num_encoder_layer=7,  # if using residual, this number must be odd
    num_decoder_layer=7,  # if using residual, this number must be odd
    num_mod1_layer=7,  # if using residual, this number must be odd
    num_mod2_layer=7,  # if using residual, this number must be odd
    epochs=1000,
    lr1=1e-2,
    lr2=1e-2,
    k1=10,
    k2=10,
    normalization='batch',
    patience=10
)

now = datetime.now()
logger = open(f'{param["logs_path"]}{now.strftime("%d_%m_%Y %H_%M_%S")}.log', 'w')
logger.write(str(args) + '\n')

logging.info('Reading `h5ad` files...')
train_mod1 = sc.read_h5ad(param['input_train_mod1'])
mod1 = train_mod1.var['feature_types'][0]
train_mod2 = sc.read_h5ad(param['input_train_mod2'])
mod2 = train_mod2.var['feature_types'][0]

if not param['use_pretrained_data']:

    # dataset_id = train_mod1.uns['dataset_id']
    input_train_mod1 = train_mod1.X
    input_train_mod2 = train_mod2.X

    mod1_train, mod1_test, mod2_train, mod2_test = train_test_split(input_train_mod1, input_train_mod2, test_size=0.1,
                                                                    random_state=args.random_seed)

    # mod1_train, mod1_reducer = embedding(mod1_train, args.mod1_feat_size, random_seed=args.random_seed)
    # mod2_train, mod2_reducer = embedding(mod2_train, args.mod2_feat_size, random_seed=args.random_seed)
    #
    # mod1_test = mod1_reducer.transform(mod1_test)
    # mod2_test = mod2_reducer.transform(mod2_test)

    mod1_test, mod1_reducer = embedding(mod1_test, args.mod1_feat_size, random_seed=args.random_seed)
    mod2_test, mod2_reducer = embedding(mod2_test, args.mod1_feat_size, random_seed=args.random_seed)

    # pk.dump(mod1_train, open(param['input_train_mod1_pretrained'], "wb"))
    pk.dump(mod1_test, open(param['input_test_mod1_pretrained'], "wb"))
    # pk.dump(mod2_train, open(param['input_train_mod2_pretrained'], "wb"))
    pk.dump(mod2_test, open(param['input_test_mod2_pretrained'], "wb"))
    pk.dump(mod1_reducer, open(param['mod1_reducer_pretrained'], "wb"))
    pk.dump(mod2_reducer, open(param['mod2_reducer_pretrained'], "wb"))
else:
    mod1_train = pk.load(open(param['input_train_mod1_pretrained'], "rb"))
    mod1_test = pk.load(open(param['input_test_mod1_pretrained'], "rb"))
    mod2_train = pk.load(open(param['input_train_mod2_pretrained'], "rb"))
    mod2_test = pk.load(open(param['input_test_mod2_pretrained'], "rb"))
    mod1_reducer = pk.load(open(param['mod1_reducer_pretrained'], "rb"))
    mod2_reducer = pk.load(open(param['mod2_reducer_pretrained'], "rb"))

# mod1_train_label = torch.from_numpy(mod1_train)
# mod2_train_label = torch.from_numpy(mod2_train)
mod1_test_label = torch.from_numpy(mod1_test)
mod2_test_label = torch.from_numpy(mod2_test)

# graph1 = knn_graph_construction(mod1_train, args.k)
# graph2 = knn_graph_construction(mod2_train, args.k)
graph1_test = knn_graph_construction(mod1_test, args.k1)
graph2_test = knn_graph_construction(mod2_test, args.k2)

net = JointGCNAutoEncoder(args)
logger.write(str(net) + '\n')

# create optimizer when reconstruct modality 1
mod1_param = []
mod1_param.extend(net.gcn1.parameters())
mod1_param.extend(net.hid_encoder.parameters())
mod1_param.extend(net.hid_decoder.parameters())
mod1_param.extend(net.mod1_decoder.parameters())
opt1 = torch.optim.Adam(mod1_param, args.lr1)

# create optimizer when reconstruct modality 2
mod2_param = []
mod2_param.extend(net.gcn2.parameters())
mod2_param.extend(net.hid_encoder.parameters())
mod2_param.extend(net.hid_decoder.parameters())
mod2_param.extend(net.mod2_decoder.parameters())
opt2 = torch.optim.Adam(mod2_param, args.lr2)

val1_loss = []
val2_loss = []
training1_loss = []
training2_loss = []
dur = []
criterion = nn.MSELoss()
trigger_times1 = 0
trigger_times2 = 0
for epoch in range(args.epochs):
    logger.write(f'epoch:  {epoch}\n')

    net.train()

    # reconstruct modality 1
    out = net(graph1_test, mod1_test_label, type='x2x', residual=True)
    loss = criterion(out, mod1_test_label)
    running_loss = loss.item()
    opt1.zero_grad()
    loss.backward()
    opt1.step()
    training1_loss.append(running_loss)
    logger.write(f'training loss 1:  {training1_loss[-1]}\n')
    logger.flush()

    # reconstruct modality 2
    out = net(graph2_test, mod2_test_label, type='u2u', residual=True)
    loss = criterion(out, mod2_test_label)
    running_loss = loss.item()
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    training2_loss.append(running_loss)
    logger.write(f'training loss 2:  {training2_loss[-1]}\n')
    logger.flush()

    # predict modality 1
    out = net(graph2_test, mod2_test_label, type='u2x', residual=True)
    loss = criterion(out, mod1_test_label)
    running_loss = loss.item()
    opt1.zero_grad()
    loss.backward()
    opt1.step()
    training1_loss.append(running_loss)
    logger.write(f'training loss 1:  {training1_loss[-1]}\n')
    logger.flush()

    # predict modality 2
    out = net(graph1_test, mod1_test_label, type='x2u', residual=True)
    loss = criterion(out, mod2_test_label)
    running_loss = loss.item()
    opt1.zero_grad()
    loss.backward()
    opt1.step()
    training1_loss.append(running_loss)
    logger.write(f'training loss 1:  {training1_loss[-1]}\n')
    logger.flush()

    # validate
    val1 = validate(net, graph1_test, mod1_test_label, type='x2x', residual=True)
    val1_loss.append(val1)
    logger.write(f'validation loss reconstruct mod1:  {val1}\n')
    logger.flush()

    val2 = validate(net, graph2_test, mod2_test_label, type='u2u', residual=True)
    val2_loss.append(val2)
    logger.write(f'validation loss reconstruct mod2:  {val2}\n')
    logger.flush()

    val3 = validate(net, graph1_test, mod2_test_label, type='x2u', residual=True)
    val1_loss.append(val1)
    logger.write(f'validation loss predict mod2:  {val1}\n')
    logger.flush()

    val4 = validate(net, graph2_test, mod1_test_label, type='u2x', residual=True)
    val1_loss.append(val1)
    logger.write(f'validation loss predict mod1:  {val1}\n')
    logger.flush()

    # calculate rmse for mod1 to mod2
    net.eval()
    with torch.no_grad():
        out = net(graph1_test, mod1_test_label, type='x2u', residual=True)
        out = out.detach().cpu()
        metric = calculate_rmse(mod2_reducer.inverse_transform(mod2_test_label),
                                mod2_reducer.inverse_transform(mod1_test_label))

    logger.write(f'rmse {mod1} to {mod2}:  {metric}\n')
    logger.flush()

    # calculate rmse for mod2 to mod1
    net.eval()
    with torch.no_grad():
        out = net(graph2_test, mod2_test_label, type='u2x', residual=True)
        out = out.detach().cpu()
        metric = calculate_rmse(mod1_reducer.inverse_transform(mod1_test_label),
                                mod1_reducer.inverse_transform(mod2_test_label))

    logger.write(f'rmse {mod2} to {mod1}:  {metric}\n')
    logger.flush()

    # early stopping
    if len(training1_loss) > 2 and training1_loss[-1] > training1_loss[-2]:
        trigger_times1 += 1
        if trigger_times1 >= args.patience:
            logger.write(f'early stopping for mod1 trigger\n')
            logger.flush()
            break
    else:
        trigger_times1 = 0

    if len(training2_loss) > 2 and training2_loss[-1] > training2_loss[-2]:
        trigger_times2 += 1
        if trigger_times2 >= args.patience:
            logger.write(f'early stopping for mod1 trigger\n')
            logger.flush()
            break
    else:
        trigger_times2 = 0

    print(epoch)
torch.save(net, f'{param["save_model_path"]}model {mod1} {mod2} {now.strftime("%d_%m_%Y %H_%M_%S")}.pkl')
