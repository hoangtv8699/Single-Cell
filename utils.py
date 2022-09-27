import torch
import dgl
import math
import scipy
import numpy as np
import pandas as pd
import scanpy as sc
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from dgl.nn.pytorch.conv import GraphConv
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import precision_score, accuracy_score, classification_report, mean_squared_error
from pytorch_metric_learning import losses
from matplotlib import pyplot as plt


def plot_loss(train_loss, val_loss):
    plt.figure()
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel('loss', fontsize=24)
    plt.xlabel('epochs', fontsize=24)
    plt.legend(['train loss', 'val loss'], loc='upper right', prop={'size': 20})
    plt.show()


def read_logs(path):
    with open(path, "r") as log_file:
        classification_train_loss = []
        classification_val_loss = []
        embed_train_loss = []
        embed_test_loss = []
        predict_train_loss = []
        predict_test_loss = []

        types = 0
        for line in log_file.readlines():
            line = line.split(' ')
            if line == 'epoch: 0':
                types += 1
            if line[0] == 'training':
                if types == 1:
                    classification_train_loss.append(float(line[2]))
                if types == 2:
                    embed_train_loss.append(float(line[2]))
                if types == 3:
                    predict_train_loss.append(float(line[2]))
            elif line[0] == 'validation':
                if types == 1:
                    classification_val_loss.append(float(line[2]))
                if types == 2:
                    embed_test_loss.append(float(line[2]))
                if types == 3:
                    predict_test_loss.append(float(line[2]))

        return classification_train_loss, classification_val_loss, embed_train_loss, embed_test_loss \
            , predict_train_loss, predict_test_loss


def embedding(mod, n_components, random_seed=0):
    # sc.pp.log1p(mod)
    # sc.pp.scale(mod)

    mod_reducer = TruncatedSVD(n_components=n_components, random_state=random_seed)
    truncated_mod = mod_reducer.fit_transform(mod)
    del mod
    return truncated_mod, mod_reducer


def knn_graph_construction(mod_train, k):
    mod_train = torch.Tensor(mod_train)
    knn_graph = dgl.knn_graph(mod_train, k)
    knn_graph = dgl.add_self_loop(knn_graph)
    return knn_graph


def validate(model, graph, labels, type='x2x', residual=True):
    model.eval()
    with torch.no_grad():
        out = model(graph, labels, type, residual)
        loss = F.mse_loss(out, labels)
        return loss


def calculate_rmse(mod, mod_answer):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(mod, mod_answer, squared=False)


class GCNlayer(nn.Module):
    def __init__(self, feat_size, hid_feats, num_layer, activation):
        super().__init__()
        self.num_layer = num_layer
        self.gcn = nn.ModuleList()
        self.gcn_acts = nn.ModuleList()
        self.gcn_norm = nn.ModuleList()

        # create gcn1 layer
        self.gcn.append(GraphConv(feat_size, hid_feats))
        for i in range(num_layer - 1):
            self.gcn.append(GraphConv(hid_feats, hid_feats))

        if activation == 'gelu':
            for i in range(num_layer):
                self.gcn_acts.append(nn.GELU())
        elif activation == 'prelu':
            for i in range(num_layer):
                self.gcn_acts.append(nn.PReLU())
        elif activation == 'relu':
            for i in range(num_layer):
                self.gcn_acts.append(nn.ReLU())
        elif activation == 'leaky_relu':
            for i in range(num_layer):
                self.gcn_acts.append(nn.LeakyReLU())

    def forward(self, graph, mod, residual=False):

        if residual:
            mod = self.gcn[0](graph, mod)
            mod = self.gcn_acts[0](mod)

            for i in range(1, int((self.num_layer - 1) / 2)):
                temp = mod
                mod = self.gcn[i * 2 - 1](graph, mod)
                mod = self.gcn_acts[i * 2 - 1](mod)

                mod = self.gcn[i * 2](graph, mod) + temp
                mod = self.gcn_acts[i * 2](mod)
        else:
            for i in range(self.num_layer):
                mod = self.gcn[i](graph, mod)
                mod = self.gcn_acts[i](mod)
        return mod


class HiddenEncoder(nn.Module):
    def __init__(self, hid_feats, latent_feats, num_layer, activation, normalization):
        super().__init__()
        self.num_layer = num_layer
        self.hid_encoder = nn.ModuleList()
        self.encoder_acts = nn.ModuleList()
        self.encoder_norm = nn.ModuleList()

        # create hidden encoder
        for i in range(num_layer - 1):
            self.hid_encoder.append(nn.Linear(hid_feats, hid_feats))
        self.hid_encoder.append(nn.Linear(hid_feats, latent_feats))

        if activation == 'gelu':
            for i in range(num_layer):
                self.encoder_acts.append(nn.GELU())
        elif activation == 'prelu':
            for i in range(num_layer):
                self.encoder_acts.append(nn.PReLU())
        elif activation == 'relu':
            for i in range(num_layer):
                self.encoder_acts.append(nn.ReLU())
        elif activation == 'leaky_relu':
            for i in range(num_layer):
                self.encoder_acts.append(nn.LeakyReLU())

        if normalization == 'batch':
            for i in range(num_layer - 1):
                self.encoder_norm.append(nn.BatchNorm1d(hid_feats))
            self.encoder_norm.append(nn.BatchNorm1d(latent_feats))
        elif normalization == 'layer':
            for i in range(num_layer - 1):
                self.encoder_norm.append(nn.LayerNorm(hid_feats))
            self.encoder_norm.append(nn.LayerNorm(latent_feats))

    def forward(self, mod, residual=False):
        if residual:
            for i in range(int((self.num_layer - 1) / 2)):
                temp = mod
                mod = self.hid_encoder[i * 2](mod)
                mod = self.encoder_norm[i * 2](mod)
                mod = self.encoder_acts[i * 2](mod)

                mod = self.hid_encoder[i * 2 + 1](mod) + temp
                mod = self.encoder_norm[i * 2 + 1](mod)
                mod = self.encoder_acts[i * 2 + 1](mod)

            mod = self.hid_encoder[-1](mod)
            mod = self.encoder_norm[-1](mod)
            mod = self.encoder_acts[-1](mod)
        else:
            for i in range(self.num_layer):
                mod = self.hid_encoder[i](mod)
                mod = self.encoder_norm[i](mod)
                mod = self.encoder_acts[i](mod)
        return mod


class HiddenDecoder(nn.Module):
    def __init__(self, hid_feats, latent_feats, num_layer, activation, normalization):
        super().__init__()
        self.num_layer = num_layer
        self.hid_decoder = nn.ModuleList()
        self.decoder_acts = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()

        # create hidden decoder
        self.hid_decoder.append(nn.Linear(latent_feats, hid_feats))
        for i in range(num_layer - 1):
            self.hid_decoder.append(nn.Linear(hid_feats, hid_feats))

        if activation == 'gelu':
            for i in range(num_layer):
                self.decoder_acts.append(nn.GELU())
        elif activation == 'prelu':
            for i in range(num_layer):
                self.decoder_acts.append(nn.PReLU())
        elif activation == 'relu':
            for i in range(num_layer):
                self.decoder_acts.append(nn.ReLU())
        elif activation == 'leaky_relu':
            for i in range(num_layer):
                self.decoder_acts.append(nn.LeakyReLU())

        if normalization == 'batch':
            # self.decoder_norm.append(nn.BatchNorm1d(latent_feats))
            for i in range(num_layer):
                self.decoder_norm.append(nn.BatchNorm1d(hid_feats))
        elif normalization == 'layer':
            # self.decoder_norm.append(nn.LayerNorm(latent_feats))
            for i in range(num_layer):
                self.decoder_norm.append(nn.LayerNorm(hid_feats))

    def forward(self, mod, residual=False):
        if residual:
            mod = self.hid_decoder[0](mod)
            mod = self.decoder_acts[0](mod)
            mod = self.decoder_norm[0](mod)

            for i in range(1, int((self.num_layer - 1) / 2)):
                temp = mod
                mod = self.hid_decoder[i * 2 - 1](mod)
                mod = self.decoder_acts[i * 2 - 1](mod)
                mod = self.decoder_norm[i * 2 - 1](mod)

                mod = self.hid_decoder[i * 2](mod) + temp
                mod = self.decoder_norm[i * 2](mod)
                mod = self.decoder_acts[i * 2](mod)
        else:
            for i in range(self.num_layer):
                mod = self.hid_decoder[i](mod)
                mod = self.decoder_norm[i](mod)
                mod = self.decoder_acts[i](mod)
        return mod


class ModalityDecoder(nn.Module):
    def __init__(self, hid_feats, out_feats, num_layer, activation, normalization):
        super().__init__()
        self.num_layer = num_layer
        self.mod_decoder = nn.ModuleList()
        self.mod_acts = nn.ModuleList()
        self.mod_norm = nn.ModuleList()

        # create modality decoder
        for i in range(num_layer - 1):
            self.mod_decoder.append(nn.Linear(hid_feats, hid_feats))
        self.mod_decoder.append(nn.Linear(hid_feats, out_feats))

        if activation == 'gelu':
            for i in range(num_layer):
                self.mod_acts.append(nn.GELU())
        elif activation == 'prelu':
            for i in range(num_layer):
                self.mod_acts.append(nn.PReLU())
        elif activation == 'relu':
            for i in range(num_layer):
                self.mod_acts.append(nn.ReLU())
        elif activation == 'leaky_relu':
            for i in range(num_layer):
                self.mod_acts.append(nn.LeakyReLU())

        if normalization == 'batch':
            for i in range(num_layer - 1):
                self.mod_norm.append(nn.BatchNorm1d(hid_feats))
            self.mod_norm.append(nn.BatchNorm1d(out_feats))
        elif normalization == 'layer':
            for i in range(num_layer - 1):
                self.mod_norm.append(nn.LayerNorm(hid_feats))
            self.mod_norm.append(nn.LayerNorm(out_feats))

    def forward(self, mod, residual=False):
        if residual:
            for i in range(int((self.num_layer - 1) / 2)):
                temp = mod
                mod = self.mod_decoder[i * 2](mod)
                mod = self.mod_acts[i * 2](mod)
                mod = self.mod_norm[i * 2](mod)

                mod = self.mod_decoder[i * 2 + 1](mod) + temp
                mod = self.mod_norm[i * 2 + 1](mod)
                mod = self.mod_acts[i * 2 + 1](mod)

            mod = self.mod_decoder[-1](mod)
            mod = self.mod_acts[-1](mod)
        else:
            for i in range(self.num_layer - 1):
                mod = self.mod_decoder[i](mod)
                mod = self.mod_norm[i](mod)
                mod = self.mod_acts[i](mod)

            mod = self.mod_decoder[-1](mod)
            mod = self.mod_acts[-1](mod)
        return mod


class JointGCNAutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # define layer
        self.gcn1 = GCNlayer(args.mod1_feat_size, args.hid_feats, args.num_gcn1_layer, args.activation)
        self.gcn2 = GCNlayer(args.mod2_feat_size, args.hid_feats, args.num_gcn2_layer, args.activation)

        self.hid_encoder = HiddenEncoder(args.hid_feats, args.latent_feats, args.num_encoder_layer, args.activation,
                                         args.normalization)
        self.hid_decoder = HiddenDecoder(args.hid_feats, args.latent_feats, args.num_decoder_layer, args.activation,
                                         args.normalization)

        self.mod1_decoder = ModalityDecoder(args.hid_feats, args.out_mod1_feats, args.num_mod1_layer, args.activation,
                                            args.normalization)
        self.mod2_decoder = ModalityDecoder(args.hid_feats, args.out_mod2_feats, args.num_mod2_layer, args.activation,
                                            args.normalization)

    def forward(self, graph, mod, type='x2u', residual=False):
        """
        :param self:
        :param graph:
            graph with data store in graph.ndata['feat']
        :param type:
            x2u = mod1 to mod2
            u2x = mod2 to mod1
            x2x = mod1 to mod1
            u2u = mod2 to mod2
        :param residual:
            using residual connection or not
        :return: reconstructed modality
        """

        if type == 'x2u':
            mod = self.gcn1(graph, mod, residual)
            mod = self.hid_encoder(mod, residual)
            mod = self.hid_decoder(mod, residual)
            mod = self.mod2_decoder(mod, residual)
        elif type == 'u2x':
            mod = self.gcn2(graph, mod, residual)
            mod = self.hid_encoder(mod, residual)
            mod = self.hid_decoder(mod, residual)
            mod = self.mod1_decoder(mod, residual)
        elif type == 'x2x':
            mod = self.gcn1(graph, mod, residual)
            mod = self.hid_encoder(mod, residual)
            mod = self.hid_decoder(mod, residual)
            mod = self.mod1_decoder(mod, residual)
        elif type == 'u2u':
            mod = self.gcn2(graph, mod, residual)
            mod = self.hid_encoder(mod, residual)
            mod = self.hid_decoder(mod, residual)
            mod = self.mod2_decoder(mod, residual)

        return mod


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


##########################################################
################ Classification model ####################
##########################################################
def cal_acc(y_true, y_pred):
    y_pred_indices = np.argmax(y_pred, axis=-1)
    n = len(y_true)
    acc = (y_pred_indices == y_true).sum().item() / n
    return acc


class ModalityDataset(Dataset):
    def __init__(self, data, labels, types='classification'):
        self.types = types

        if self.types == 'classification':
            self.data = data.toarray()
            self.labels = labels
        elif self.types == '2mod':
            self.data = data.toarray()
            self.labels = labels.toarray()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cell = torch.tensor(self.data[idx]).float()
        label = 0

        if self.types == 'classification':
            label = torch.tensor(self.labels[idx]).long()
        elif self.types == '2mod':
            label = torch.tensor(self.labels[idx]).float()

        return cell, label


class ModalityDataset2(Dataset):
    def __init__(self, adata1, adata2, types='classification'):
        self.types = types
        self.adata1 = adata1
        self.adata2 = adata2

    def __len__(self):
        return self.adata1.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = torch.tensor(self.adata1.X[idx].toarray()[0]).float()
        y = 0
        if self.types == 'classification':
            y = torch.tensor(self.adata2[idx]).long()
        elif self.types == '2mod':
            y = torch.tensor(self.adata2.X[idx].toarray()[0]).float()

        return X, y


class CellClassification(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_layer = args.num_layer
        self.hid_layer = nn.ModuleList()
        self.layer_acts = nn.ModuleList()
        self.layer_norm = nn.ModuleList()

        # create hidden decoder
        self.hid_layer.append(nn.Linear(args.input_feats, args.hid_feats))
        for i in range(args.num_layer - 1):
            self.hid_layer.append(nn.Linear(args.hid_feats, args.hid_feats))
        self.hid_layer.append(nn.Linear(args.hid_feats, args.num_class))

        if args.activation == 'gelu':
            for i in range(args.num_layer):
                self.layer_acts.append(nn.GELU())
        elif args.activation == 'prelu':
            for i in range(args.num_layer):
                self.layer_acts.append(nn.PReLU())
        elif args.activation == 'relu':
            for i in range(args.num_layer):
                self.layer_acts.append(nn.ReLU())
        elif args.activation == 'leaky_relu':
            for i in range(args.num_layer):
                self.layer_acts.append(nn.LeakyReLU())
        self.layer_acts.append(nn.Softmax(dim=1))

        if args.normalization == 'batch':
            for i in range(args.num_layer):
                self.layer_norm.append(nn.BatchNorm1d(args.hid_feats))
        elif args.normalization == 'layer':
            for i in range(args.num_layer):
                self.layer_norm.append(nn.LayerNorm(args.hid_feats))

        if args.dropout:
            self.dropout = nn.Dropout(args.dropout)

    def forward(self, mod, residual=False):
        if residual:
            mod = self.hid_layer[0](mod)
            mod = self.layer_acts[0](mod)
            mod = self.layer_norm[0](mod)
            if self.dropout:
                mod = self.dropout(mod)

            for i in range(1, int((self.num_layer - 1) / 2)):
                temp = mod
                mod = self.hid_layer[i * 2 - 1](mod)
                mod = self.layer_acts[i * 2 - 1](mod)
                mod = self.layer_norm[i * 2 - 1](mod)
                if self.dropout:
                    mod = self.dropout(mod)

                mod = self.hid_layer[i * 2](mod) + temp
                mod = self.layer_acts[i * 2](mod)
                mod = self.layer_norm[i * 2](mod)
                if self.dropout:
                    mod = self.dropout(mod)

            mod = self.hid_layer[-1](mod)
            mod = self.layer_acts[-1](mod)
        else:
            for i in range(self.num_layer):
                mod = self.hid_layer[i](mod)
                mod = self.layer_acts[i](mod)
                mod = self.layer_norm[i](mod)
                if self.dropout:
                    mod = self.dropout(mod)

            mod = self.hid_layer[-1](mod)
            mod = self.layer_acts[-1](mod)
        return mod


##########################################################
################ Contrastive learning ####################
##########################################################
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


# train model as classification
def train_classification(train_loader, val_loader, net, args, logger):
    print('train classification')
    net.cuda()
    net_param = []
    net_param.extend(net.embed.parameters())
    net_param.extend(net.classification.parameters())
    opt = torch.optim.Adam(net_param, args.lr_classification)

    training_loss = []
    val_loss = []
    criterion = nn.CrossEntropyLoss()
    trigger_times = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_loss = 0
        for train_batch, label in train_loader:
            train_batch, label = train_batch.cuda(), label.cuda()

            opt.zero_grad()
            out = net(train_batch, residual=True, types='classification')
            loss = criterion(out, label)
            running_loss += loss.item() * train_batch.size(0)
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss: {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        y_pred = []
        y_true = []
        with torch.no_grad():
            for val_batch, label in val_loader:
                val_batch, label = val_batch.cuda(), label.cuda()

                out = net(val_batch, residual=True, types='classification')
                loss = criterion(out, label)
                running_loss += loss.item() * val_batch.size(0)
                if len(y_pred) == 0:
                    y_pred = out
                    y_true = label
                else:
                    y_pred = torch.cat((y_pred, out), dim=0)
                    y_true = torch.cat((y_true, label), dim=0)

            acc = cal_acc(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
            val_loss.append(running_loss / len(val_loader.dataset))
            report = classification_report(y_true.detach().cpu().numpy(),
                                           np.argmax(y_pred.detach().cpu().numpy(), axis=-1),
                                           target_names=args.classes_, zero_division=1)
        logger.write(f'validation loss: {val_loss[-1]}\n')
        logger.write(f'validation acc: {acc}\n')
        logger.write(f'classification report: \n{report}\n')
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


# train contrastive
def train_contrastive(train_loader, val_loader, net1, net2, args1, logger):
    print('train contrastive')
    net1.cuda()
    net2.cuda()

    opt = torch.optim.Adam(list(net1.embed.parameters()) + list(net2.embed.parameters()), args1.lr_embed)

    training_loss = []
    val_loss = []
    criterion = losses.NTXentLoss(temperature=0.10)
    trigger_times = 0
    best_loss = 10000
    best_state_dict1 = net1.state_dict()
    best_state_dict2 = net2.state_dict()

    for epoch in range(args1.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net1.train()
        net2.train()
        running_loss = 0
        for mod1_batch, mod2_batch in train_loader:
            mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

            opt.zero_grad()
            out1 = net1(mod1_batch, residual=True, types='embed')
            out2 = net2(mod2_batch, residual=True, types='embed')

            out = torch.cat((out1, out2))
            indices = torch.arange(0, out1.size(0), device=out1.device)
            labels = torch.cat((indices, indices))

            loss = criterion(out, labels)
            running_loss += loss.item() * mod1_batch.size(0)
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss: {training_loss[-1]}\n')
        logger.flush()

        # validating
        net1.eval()
        net2.eval()
        running_loss = 0
        with torch.no_grad():
            for mod1_batch, mod2_batch in val_loader:
                mod1_batch, mod2_batch = mod1_batch.cuda(), mod2_batch.cuda()

                out1 = net1(mod1_batch, residual=True, types='embed')
                out2 = net2(mod2_batch, residual=True, types='embed')

                out = torch.cat((out1, out2))
                indices = torch.arange(0, out1.size(0), device=out1.device)
                labels = torch.cat((indices, indices))

                loss = criterion(out, labels)
                running_loss += loss.item() * mod1_batch.size(0)

            val_loss.append(running_loss / len(val_loader.dataset))
        logger.write(f'validation loss: {val_loss[-1]}\n')
        logger.flush()

        # early stopping
        if len(val_loss) > 2 and val_loss[-1] >= best_loss:
            trigger_times += 1
            if trigger_times >= args1.patience:
                logger.write(f'early stopping because val loss not decrease for {args1.patience} epoch\n')
                logger.flush()
                break
        else:
            best_loss = val_loss[-1]
            best_state_dict1 = net1.state_dict()
            best_state_dict2 = net2.state_dict()
            trigger_times = 0

        print(epoch)
    return best_state_dict1, best_state_dict2


# train model to predict modality
def train_predict(train_loader, val_loader, net, args, logger, mod_reducer):
    print('train predict')
    net.cuda()
    net_param = []
    net_param.extend(net.predict.parameters())
    opt = torch.optim.Adam(net_param, args.lr_predict)

    training_loss = []
    val_loss = []
    rmse = []
    criterion = nn.MSELoss()
    trigger_times = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_loss = 0
        for train_batch, label in train_loader:
            train_batch, label = train_batch.cuda(), label.cuda()

            opt.zero_grad()
            out = net(train_batch, residual=True, types='predict')
            loss = criterion(out, label)
            running_loss += loss.item() * train_batch.size(0)
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss:  {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        running_rmse = 0
        with torch.no_grad():
            for val_batch, label in val_loader:
                val_batch, label = val_batch.cuda(), label.cuda()

                out = net(val_batch, residual=True, types='predict')
                loss = criterion(out, label)
                running_loss += loss.item() * val_batch.size(0)
                # using reducer
                if mod_reducer:
                    out_ori = mod_reducer.inverse_transform(out.detach().cpu().numpy())
                    label_ori = mod_reducer.inverse_transform(label.detach().cpu().numpy())
                    running_rmse += mean_squared_error(label_ori, out_ori) * val_batch.size(0)
                else:
                    running_rmse += mean_squared_error(label.detach().cpu().numpy(),
                                                       out.detach().cpu().numpy()) * val_batch.size(0)

            val_loss.append(running_loss / len(val_loader.dataset))
            rmse.append(math.sqrt(running_rmse / len(val_loader.dataset)))
        logger.write(f'validation loss: {val_loss[-1]}\n')
        logger.write(f'validation rmse: {rmse[-1]}\n')
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


class ResidualBlock(nn.Module):
    def __init__(self, input_feats, activation, normalization, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.layers.append(nn.Linear(input_feats, input_feats))
        self.layers.append(nn.Linear(input_feats, input_feats))

        if activation == 'gelu':
            self.acts.append(nn.GELU())
            self.acts.append(nn.GELU())
        elif activation == 'prelu':
            self.acts.append(nn.PReLU())
            self.acts.append(nn.PReLU())
        elif activation == 'relu':
            self.acts.append(nn.ReLU())
            self.acts.append(nn.ReLU())
        elif activation == 'leaky_relu':
            self.acts.append(nn.LeakyReLU())
            self.acts.append(nn.LeakyReLU())

        if normalization == 'batch':
            self.norms.append(nn.BatchNorm1d(input_feats))
            self.norms.append(nn.BatchNorm1d(input_feats))
        elif normalization == 'layer':
            self.norms.append(nn.LayerNorm(input_feats))
            self.norms.append(nn.LayerNorm(input_feats))

    def forward(self, x):
        temp = x
        x = self.layers[0](x)
        x = self.acts[0](x)
        x = self.norms[0](x)
        x = self.dropout(x)

        x = self.layers[1](x) + temp
        x = self.acts[1](x)
        x = self.norms[1](x)
        x = self.dropout(x)

        return x


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
            for i in range(self.num_layer):
                mod = self.hid_layer[i](mod)
                mod = self.layer_acts[i](mod)
                mod = self.layer_norm[i](mod)
                mod = self.dropout(mod)

            mod = self.hid_layer[-1](mod)
            mod = self.layer_acts[-1](mod)
        return mod


class ContrastiveModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = AbsModel(args.input_feats, args.embed_hid_feats, args.latent_feats, args.num_embed_layer,
                              args.activation, args.normalization, args.dropout, 'relu')
        self.classification = AbsModel(args.latent_feats, args.class_hid_feats, args.num_class, args.num_class_layer,
                                       args.activation, args.normalization, args.dropout, 'softmax')
        self.predict = AbsModel(args.latent_feats, args.pred_hid_feats, args.out_feats, args.num_pred_layer,
                                args.activation, args.normalization, args.dropout, args.act_out)

    def forward(self, mod, residual=False, types="class"):
        if types == 'classification':
            # train classification
            mod = self.embed(mod, residual)
            mod = self.classification(mod, residual)
        elif types == 'embed':
            # train contrastive learning
            mod = self.embed(mod, residual)
        elif types == 'predict':
            # train predicting modality
            mod = self.embed(mod, residual)
            mod = self.predict(mod, residual)
        return mod


def drop_data(adata):
    # del adata.obs.drop(adata.obs.index, inplace=True)
    # del adata.var.drop(adata.var.index, inplace=True)
    del adata.obs
    del adata.var
    del adata.uns
    del adata.obsm
    del adata.layers
    return adata


class LinearRegressionModel(torch.nn.Module):

    def __init__(self, int, out):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(int, out)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
