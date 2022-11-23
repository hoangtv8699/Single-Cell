import math

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from pytorch_metric_learning import losses
from scipy.stats import pearsonr
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import Dataset


def plot_loss(train_loss, val_loss):
    plt.figure()
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epochs', fontsize=20)
    plt.legend(['train loss', 'val loss'], loc='upper right', prop={'size': 20})
    plt.show()


def read_logs(path):
    with open(path, "r") as log_file:
        classification_train_loss1 = []
        classification_val_loss1 = []
        classification_train_loss2 = []
        classification_val_loss2 = []
        embed_train_loss = []
        embed_test_loss = []
        predict_train_loss1 = []
        predict_test_loss1 = []
        predict_train_loss2 = []
        predict_test_loss2 = []

        i = 0
        types = 0
        for line in log_file.readlines():
            if i == 180:
                a = 10
            if line == 'epoch:  0\n':
                types += 1
            line = line.split(' ')
            if line[0] == 'training' and line[1] == 'loss:':
                if types == 1:
                    classification_train_loss1.append(float(line[2]))
                elif types == 2:
                    classification_train_loss2.append(float(line[2]))
                elif types == 3:
                    embed_train_loss.append(float(line[2]))
                elif types == 4:
                    predict_train_loss1.append(float(line[3]))
                elif types == 5:
                    predict_train_loss2.append(float(line[3]))
            elif line[0] == 'validation' and line[1] == 'loss:':
                if types == 1:
                    classification_val_loss1.append(float(line[2]))
                elif types == 2:
                    classification_val_loss2.append(float(line[2]))
                elif types == 3:
                    embed_test_loss.append(float(line[2]))
                elif types == 4:
                    predict_test_loss1.append(float(line[2]))
                elif types == 5:
                    predict_test_loss2.append(float(line[2]))
            i += 1

        return classification_train_loss1, classification_val_loss1, classification_train_loss2, classification_val_loss2, \
               embed_train_loss, embed_test_loss, predict_train_loss1, predict_test_loss1, predict_train_loss2, predict_test_loss2


def embedding(mod, n_components, random_seed=0):
    # sc.pp.log1p(mod)
    # sc.pp.scale(mod)

    mod_reducer = TruncatedSVD(n_components=n_components, random_state=random_seed)
    truncated_mod = mod_reducer.fit_transform(mod)
    del mod
    return truncated_mod, mod_reducer


def validate(model, graph, labels, type='x2x', residual=True):
    model.eval()
    with torch.no_grad():
        out = model(graph, labels, type, residual)
        loss = f.mse_loss(out, labels)
        return loss


def calculate_rmse(mod, mod_answer):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(mod, mod_answer, squared=False)


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


def cal_acc(y_true, y_pred):
    y_pred_indices = np.argmax(y_pred, axis=-1)
    n = len(y_true)
    acc = (y_pred_indices == y_true).sum().item() / n
    return acc


class ModalityDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cell = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).float()
        return cell, label


class ModalityDataset2(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cell = torch.tensor(self.data.X[idx].toarray()[0]).float()
        label = torch.tensor(self.labels.X[idx].toarray()[0]).float()
        return cell, label


# ranking feature
def analysis_features(adata, method='wilcoxon', top=100):
    adata_cpm = adata.copy()
    adata_cpm.X = adata_cpm.layers['counts']
    sc.pp.log1p(adata_cpm)
    sc.tl.rank_genes_groups(adata_cpm, groupby='cell_type', method=method)

    cell_types = adata_cpm.obs.cell_type.value_counts().index

    df = pd.DataFrame()
    for cell_type in cell_types:
        dedf = sc.get.rank_genes_groups_df(adata_cpm, group=cell_type)
        dedf['cell_type'] = cell_type
        dedf = dedf.sort_values('scores', ascending=False)
        dedf = dedf[dedf['scores'] > 0].iloc[:top]
        df = df.append(dedf, ignore_index=True)
    selected_genes = set(df.names)
    subset = selected_genes.intersection(adata_cpm.var_names)
    del adata_cpm
    return subset


# train contrastive
def train_contrastive(train_loader, val_loader, net, args, logger):
    print('train contrastive')
    net.cuda()
    net_param = list(net.input1.parameters()) + list(net.input2.parameters()) + list(net.encoder.parameters())
    opt = torch.optim.Adam(net_param, args.lr_contras)

    training_contras_loss = []
    val_contras_loss = []
    criterion = losses.NTXentLoss(temperature=0.10)
    trigger_times = 0
    best_contras = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_contras_loss = 0
        for mod1_batch, mod2_batch in train_loader:
            mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

            opt.zero_grad()
            # optimize encoder contrastive
            embed1 = net(mod1_batch, types='embed1')
            embed2 = net(mod2_batch, types='embed2')
            embed = torch.cat((embed1, embed2))
            indices = torch.arange(0, embed1.size(0), device=embed1.device)
            labels = torch.cat((indices, indices))
            loss = criterion(embed, labels)
            running_contras_loss += loss.item() * mod1_batch.size(0)
            loss.backward()
            opt.step()

        training_contras_loss.append(running_contras_loss / len(train_loader.dataset))
        logger.write(f'training contras loss: {training_contras_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_contras_loss = 0
        with torch.no_grad():
            for mod1_batch, mod2_batch in val_loader:
                mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

                # optimize encoder contrastive
                embed1 = net(mod1_batch, types='embed1')
                embed2 = net(mod2_batch, types='embed2')
                embed = torch.cat((embed1, embed2))
                indices = torch.arange(0, embed1.size(0), device=embed1.device)
                labels = torch.cat((indices, indices))
                loss = criterion(embed, labels)
                running_contras_loss += loss.item() * mod1_batch.size(0)

            val_contras_loss.append(running_contras_loss / len(val_loader.dataset))
            logger.write(f'val contras loss: {val_contras_loss[-1]}\n')
            logger.flush()

        # early stopping
        if len(val_contras_loss) > 2 and val_contras_loss[-1] >= best_contras:
            trigger_times += 1
            if trigger_times >= args.patience:
                logger.write(f'early stopping because val loss not decrease for {args.patience} epoch\n')
                logger.flush()
                break
        else:
            best_contras = val_contras_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0

        print(epoch)
    return best_state_dict


# train autoencoder
def train_autoencoder(train_loader, val_loader, net, args, logger):
    print('train autoencoder')
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr_ae)

    training_recon1_loss = []
    training_recon2_loss = []
    val_recon1_loss = []
    val_recon2_loss = []
    criterion = nn.MSELoss()
    trigger_times = 0
    best_loss1 = 10000
    best_loss2 = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_recon1_loss = 0
        running_recon2_loss = 0
        for mod1_batch, mod2_batch in train_loader:
            mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

            opt.zero_grad()
            out1 = net(mod1_batch, types='1to1')
            out2 = net(mod2_batch, types='2to2')
            loss1 = criterion(out1, mod1_batch)
            loss2 = criterion(out2, mod2_batch)
            running_recon1_loss += loss1.item() * mod1_batch.size(0)
            running_recon2_loss += loss2.item() * mod2_batch.size(0)
            loss1.backward()
            loss2.backward()
            opt.step()

        training_recon1_loss.append(running_recon1_loss / len(train_loader.dataset))
        training_recon2_loss.append(running_recon2_loss / len(train_loader.dataset))
        logger.write(f'training recon1 loss: {training_recon1_loss[-1]}\n')
        logger.write(f'training recon2 loss: {training_recon2_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_recon1_loss = 0
        running_recon2_loss = 0
        with torch.no_grad():
            for mod1_batch, mod2_batch in val_loader:
                mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

                # optimize encoder contrastive
                out1 = net(mod1_batch, types='1to1')
                out2 = net(mod2_batch, types='2to2')
                loss1 = criterion(out1, mod1_batch)
                loss2 = criterion(out2, mod2_batch)
                running_recon1_loss += loss1.item() * mod1_batch.size(0)
                running_recon2_loss += loss2.item() * mod2_batch.size(0)

            val_recon1_loss.append(running_recon1_loss / len(val_loader.dataset))
            val_recon2_loss.append(running_recon2_loss / len(val_loader.dataset))
            logger.write(f'val recon1 loss: {val_recon1_loss[-1]}\n')
            logger.write(f'val recon2 loss: {val_recon2_loss[-1]}\n')
            logger.flush()

        # early stopping
        # if len(val_recon1_loss) > 2 and (val_recon1_loss[-1] >= best_loss1 and val_recon2_loss[-1] >= best_loss2):
        if len(val_recon1_loss) > 2 and val_recon1_loss[-1] >= best_loss1:
            trigger_times += 1
            if trigger_times >= args.patience:
                logger.write(f'early stopping because val loss not decrease for {args.patience} epoch\n')
                logger.flush()
                break
        else:
            best_loss1 = val_recon1_loss[-1]
            best_loss2 = val_recon2_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0

        print(epoch)
    return best_state_dict


# train model to predict modality
def train_predict(train_loader, val_loader, net, args, logger):
    print('train predict')
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr_pred)

    training_recon1_loss = []
    training_recon2_loss = []
    val_recon1_loss = []
    val_recon2_loss = []
    criterion = nn.MSELoss()
    trigger_times = 0
    best_loss1 = 10000
    best_loss2 = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_recon1_loss = 0
        running_recon2_loss = 0
        for mod1_batch, mod2_batch in train_loader:
            mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

            opt.zero_grad()
            out1 = net(mod1_batch, types='1to2')
            out2 = net(mod2_batch, types='2to1')
            loss1 = criterion(out1, mod2_batch)
            loss2 = criterion(out2, mod1_batch)
            running_recon1_loss += loss1.item() * mod1_batch.size(0)
            running_recon2_loss += loss2.item() * mod2_batch.size(0)
            loss1.backward()
            loss2.backward()
            opt.step()

        training_recon1_loss.append(running_recon1_loss / len(train_loader.dataset))
        training_recon2_loss.append(running_recon2_loss / len(train_loader.dataset))
        logger.write(f'training predict 1to2 loss: {training_recon1_loss[-1]}\n')
        logger.write(f'training predict 2to1 loss: {training_recon2_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_recon1_loss = 0
        running_recon2_loss = 0
        with torch.no_grad():
            for mod1_batch, mod2_batch in val_loader:
                mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

                # optimize encoder contrastive
                out1 = net(mod1_batch, types='1to2')
                out2 = net(mod2_batch, types='2to1')
                loss1 = criterion(out1, mod2_batch)
                loss2 = criterion(out2, mod1_batch)
                running_recon1_loss += loss1.item() * mod1_batch.size(0)
                running_recon2_loss += loss2.item() * mod2_batch.size(0)

            val_recon1_loss.append(running_recon1_loss / len(val_loader.dataset))
            val_recon2_loss.append(running_recon2_loss / len(val_loader.dataset))
            logger.write(f'val predict 1to2 loss: {val_recon1_loss[-1]}\n')
            logger.write(f'val predict 2to1 loss: {val_recon2_loss[-1]}\n')
            logger.flush()

        # early stopping
        # if len(val_recon1_loss) > 2 and (val_recon1_loss[-1] >= best_loss1 and val_recon2_loss[-1] >= best_loss2):
        if len(val_recon1_loss) > 2 and val_recon1_loss[-1] >= best_loss1:
            trigger_times += 1
            if trigger_times >= args.patience:
                logger.write(f'early stopping because val loss not decrease for {args.patience} epoch\n')
                logger.flush()
                break
        else:
            best_loss1 = val_recon1_loss[-1]
            best_loss2 = val_recon2_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0

        print(epoch)
    return best_state_dict


def train_linear(train_loader, val_loader, net, args, logger):
    print('train linear')
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr)

    training_loss = []
    val_loss = []
    criterion = nn.MSELoss()
    trigger_times = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_loss = 0
        for mod1_batch, mod2_batch in train_loader:
            mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

            opt.zero_grad()
            out = net(mod1_batch)
            loss = criterion(out, mod2_batch)
            running_loss += loss.item() * mod1_batch.size(0)
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss: {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        with torch.no_grad():
            for mod1_batch, mod2_batch in val_loader:
                mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

                # optimize encoder contrastive
                out = net(mod1_batch)
                loss = criterion(out, mod2_batch)
                running_loss += loss.item() * mod1_batch.size(0)

            val_loss.append(running_loss / len(val_loader.dataset))
            logger.write(f'val loss: {val_loss[-1]}\n')
            logger.flush()

        # early stopping
        if len(val_loss) > 2 and val_loss[-1] >= best_loss:
            trigger_times += 1
            if trigger_times >= args.patience:
                logger.write(f'early stopping because val loss not decrease for {args.patience} epoch\n')
                logger.flush()
                break
        else:
            best_loss = val_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0

        print(epoch)
    return best_state_dict


class AbsModel(nn.Module):
    def __init__(self, input_feats, hid_feats, latent_feats, num_layer, activation, normalization, dropout,
                 act_out):
        super().__init__()
        self.num_layer = num_layer
        self.hid_layer = nn.ModuleList()
        self.layer_acts = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # create hidden decoder
        self.hid_layer.append(nn.Linear(input_feats, hid_feats))
        for i in range(num_layer - 2):
            self.hid_layer.append(nn.Linear(hid_feats, hid_feats))
        self.hid_layer.append(nn.Linear(hid_feats, latent_feats))

        if activation == 'gelu':
            for i in range(num_layer - 1):
                self.layer_acts.append(nn.GELU())
        elif activation == 'prelu':
            for i in range(num_layer - 1):
                self.layer_acts.append(nn.PReLU())
        elif activation == 'relu':
            for i in range(num_layer - 1):
                self.layer_acts.append(nn.ReLU())
        elif activation == 'leaky_relu':
            for i in range(num_layer - 1):
                self.layer_acts.append(nn.LeakyReLU())
        if act_out == 'softmax':
            self.layer_acts.append(nn.Softmax(dim=1))
        elif act_out == 'relu':
            self.layer_acts.append(nn.ReLU())
        elif act_out == 'sigmoid':
            self.layer_acts.append(nn.Sigmoid())

        if normalization == 'batch':
            for i in range(num_layer - 1):
                self.layer_norm.append(nn.BatchNorm1d(hid_feats))
        elif normalization == 'layer':
            for i in range(num_layer - 1):
                self.layer_norm.append(nn.LayerNorm(hid_feats))

    def forward(self, mod, residual=False):
        if residual:
            mod = self.hid_layer[0](mod)
            mod = self.layer_acts[0](mod)
            mod = self.layer_norm[0](mod)
            mod = self.dropout(mod)

            for i in range(1, int((self.num_layer - 1) / 2)):
                temp = mod
                mod = self.hid_layer[i * 2 - 1](mod)
                mod = self.layer_acts[i * 2 - 1](mod)
                mod = self.layer_norm[i * 2 - 1](mod)
                mod = self.dropout(mod)

                mod = self.hid_layer[i * 2](mod) + temp
                mod = self.layer_acts[i * 2](mod)
                mod = self.layer_norm[i * 2](mod)
                mod = self.dropout(mod)

            mod = self.hid_layer[-1](mod)
            mod = self.layer_acts[-1](mod)
        else:
            for i in range(self.num_layer - 1):
                mod = self.hid_layer[i](mod)
                mod = self.layer_acts[i](mod)
                mod = self.layer_norm[i](mod)
                mod = self.dropout(mod)

            mod = self.hid_layer[-1](mod)
            mod = self.layer_acts[-1](mod)
        return mod


###### multi-head attention encoder ######
class Encoder(nn.Module):
    def __init__(self, dim_inp, dropout, attention_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(1, num_heads=attention_heads, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(dim_inp, dim_inp),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim_inp, dim_inp),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim_inp)

    def forward(self, x, attention_mask):
        x_ori = x
        x = torch.unsqueeze(x, -1)
        res = torch.squeeze(self.attention(x, x, x, key_padding_mask=attention_mask)[0], -1)
        res = self.norm(x_ori + res)
        res = self.norm(res + self.fc(res))
        return res


class BERT(nn.Module):
    def __init__(self, dim_inp, dim_out, num_layer, attention_heads=1, dropout=0.1):
        super(BERT, self).__init__()
        self.encoder = nn.ModuleList([
            Encoder(dim_inp, dropout, attention_heads) for _ in range(num_layer)
        ])
        self.fc = nn.Linear(dim_inp, dim_out)

    def forward(self, input_tensor, attention_mask=None):
        for head in self.encoder:
            input_tensor = head(input_tensor, attention_mask)
        input_tensor = self.fc(input_tensor)
        return input_tensor


class ContrastiveModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.embed = BERT(args.input_feats, args.latent_feats, args.num_embed_layer, dropout=args.dropout)
        self.embed = AbsModel(args.input_feats, args.embed_hid_feats, args.latent_feats, args.num_embed_layer,
                              args.activation, args.normalization, args.dropout, args.act_out)
        self.predict = AbsModel(args.latent_feats, args.pred_hid_feats, args.out_feats, args.num_pred_layer,
                                args.activation, args.normalization, args.dropout, args.act_out)

    def forward(self, mod, residual=False, types="embed"):
        if types == 'embed':
            # train contrastive learning
            mod = self.embed(mod, residual)
        elif types == 'predict':
            # train predicting modality
            mod = self.embed(mod, residual)
            mod = self.predict(mod, residual)
        return mod


class ContrastiveModel2(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.embed = BERT(args.input_feats, args.latent_feats, args.num_embed_layer, dropout=args.dropout)
        self.input1 = AbsModel(args.input_feats1, args.embed_hid_feats, args.embed_hid_feats, args.num_embed_layer,
                               args.activation, args.normalization, args.dropout, args.act_out)
        self.input2 = AbsModel(args.input_feats2, args.embed_hid_feats, args.embed_hid_feats, args.num_embed_layer,
                               args.activation, args.normalization, args.dropout, args.act_out)
        self.encoder = AbsModel(args.embed_hid_feats, args.embed_hid_feats, args.latent_feats, args.num_embed_layer,
                                args.activation, args.normalization, args.dropout, args.act_out)
        self.decoder = AbsModel(args.latent_feats, args.embed_hid_feats, args.embed_hid_feats, args.num_embed_layer,
                                args.activation, args.normalization, args.dropout, args.act_out)
        self.predict1 = AbsModel(args.embed_hid_feats, args.pred_hid_feats, args.out_feats1, args.num_pred_layer,
                                 args.activation, args.normalization, args.dropout, args.act_out)
        self.predict2 = AbsModel(args.embed_hid_feats, args.pred_hid_feats, args.out_feats2, args.num_pred_layer,
                                 args.activation, args.normalization, args.dropout, args.act_out)

    def forward(self, mod, residual=False, types="embed1"):
        if types == 'embed1':
            mod = self.input1(mod, residual)
            mod = self.encoder(mod, residual)
        elif types == 'embed2':
            mod = self.input2(mod, residual)
            mod = self.encoder(mod, residual)
        elif types == 'pred1':
            mod = self.decoder(mod, residual)
            mod = self.predict1(mod, residual)
        elif types == 'pred2':
            mod = self.decoder(mod, residual)
            mod = self.predict2(mod, residual)
        elif types == '1to1':
            mod = self.input1(mod, residual)
            mod = self.encoder(mod, residual)
            mod = self.decoder(mod, residual)
            mod = self.predict1(mod, residual)
        elif types == '2to2':
            mod = self.input2(mod, residual)
            mod = self.encoder(mod, residual)
            mod = self.decoder(mod, residual)
            mod = self.predict2(mod, residual)
        elif types == '1to2':
            mod = self.input1(mod, residual)
            mod = self.encoder(mod, residual)
            mod = self.decoder(mod, residual)
            mod = self.predict2(mod, residual)
        elif types == '2to1':
            mod = self.input2(mod, residual)
            mod = self.encoder(mod, residual)
            mod = self.decoder(mod, residual)
            mod = self.predict1(mod, residual)
        return mod


class LinearRegressionModel(torch.nn.Module):

    def __init__(self, input_ft, output_ft):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_ft, output_ft)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
