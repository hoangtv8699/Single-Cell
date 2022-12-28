import os
import pickle as pk
import threading

import math
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as f
from matplotlib import pyplot as plt
from pytorch_metric_learning import losses
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def validate(model, graph, labels, type='x2x', residual=True):
    model.eval()
    with torch.no_grad():
        out = model(graph, labels, type, residual)
        loss = f.mse_loss(out, labels)
        return loss


def calculate_rmse(mod, mod_answer):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(mod, mod_answer, squared=False)


class ModalityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cell = torch.tensor(self.data[idx].X.toarray()[0]).float()
        cell_type = torch.tensor(self.data[idx].obs['cell_type_num']).long()
        # batch = torch.tensor(self.data[idx].obs['batch_num']).float()

        return cell, cell_type


class ModalityDataset2(Dataset):
    def __init__(self, mod1, mod2):
        self.mod1 = mod1
        self.mod2 = mod2

    def __len__(self):
        return self.mod1.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mod1 = torch.tensor(self.mod1[idx]).float()
        mod2 = torch.tensor(self.mod2[idx]).float()

        return mod1, mod2


def train_contrastive(train_loader, val_loader, net, args, logger):
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr_contras)

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
        for cells, cell_types in train_loader:
            labels = []
            labels_dict = {}
            i = 0
            for cell_type in cell_types:
                if str(cell_type) not in labels_dict.keys():
                    labels_dict[str(cell_type)] = i
                    labels.append(i)
                else:
                    labels.append(labels_dict[str(cell_type)])
                i += 1

            labels = torch.tensor(labels).long().cuda()
            cells, cell_types = cells.cuda(), cell_types.cuda()

            # optimize encoder contrastive
            opt.zero_grad()
            out = net(cells, cell_types)
            loss = criterion(out, labels)
            running_contras_loss += loss.item() * cells.size(0)
            loss.backward()
            opt.step()

        training_contras_loss.append(running_contras_loss / len(train_loader.dataset))
        logger.write(f'training contras loss: {training_contras_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_contras_loss = 0
        with torch.no_grad():
            for cells, cell_types in val_loader:
                labels = []
                labels_dict = {}
                i = 0
                for cell_type in cell_types:
                    if str(cell_type) not in labels_dict.keys():
                        labels_dict[str(cell_type)] = i
                        labels.append(i)
                    else:
                        labels.append(labels_dict[str(cell_type)])
                    i += 1

                labels = torch.tensor(labels).long().cuda()
                cells, cell_types = cells.cuda(), cell_types.cuda()

                out = net(cells, cell_types)
                loss = criterion(out, labels)
                running_contras_loss += loss.item() * cells.size(0)

            val_contras_loss.append(running_contras_loss / len(val_loader.dataset))
            logger.write(f'val contras loss: {val_contras_loss[-1]}\n')
            logger.flush()

        # early stopping
        if len(val_contras_loss) > 2 and round(val_contras_loss[-1], 3) >= round(best_contras, 3):
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


def train_predict(train_loader, val_loader, net, args, logger):
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr_contras)

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
        for cells, labels in train_loader:
            cells, labels = cells.cuda(), labels.cuda()

            # optimize encoder contrastive
            opt.zero_grad()
            out = net(cells)
            loss = criterion(out, labels)
            running_loss += loss.item() * cells.size(0)
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss: {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        with torch.no_grad():
            for cells, labels in val_loader:
                cells, labels = cells.cuda(), labels.cuda()

                # optimize encoder contrastive
                opt.zero_grad()
                out = net(cells)
                loss = criterion(out, labels)
                running_loss += loss.item() * cells.size(0)

        val_loss.append(running_loss / len(val_loader.dataset))
        logger.write(f'val loss: {val_loss[-1]}\n')
        logger.flush()

        # early stopping
        if len(val_loss) > 2 and round(val_loss[-1], 3) >= round(best_loss, 4):
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


class ContrastiveEmbed(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.cell_types_embed = torch.nn.Embedding(22, 8)
        # self.batch_embed = torch.nn.Embedding(13, 8)
        self.atac_embed = torch.nn.Linear(input_size, 8)
        self.linears = torch.nn.ModuleList()

        self.linears.append(torch.nn.Linear(16, 16))
        self.linears.append(torch.nn.Linear(16, 16))
        self.linears.append(torch.nn.Linear(16, 8))

    def forward(self, atac, cell_type):
        # embed atac and embed cell_type
        atac = f.relu(self.atac_embed(atac))
        cell_type = self.cell_types_embed(cell_type)
        cell_type = torch.squeeze(cell_type)

        x = torch.cat((atac, cell_type), 1)
        for linear in self.linears:
            x = f.relu(linear(x))
        return x


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.ModuleList()

        self.linears.append(torch.nn.Linear(8, 16))
        self.linears.append(torch.nn.Linear(16, 16))
        self.linears.append(torch.nn.Linear(16, 1))

    def forward(self, x):
        for linear in self.linears:
            x = f.relu(linear(x))
        return x


def cal_rmse(ad_pred, ad_sol):
    tmp = ad_sol - ad_pred
    rmse = np.sqrt(tmp.power(2).mean())
    return rmse
