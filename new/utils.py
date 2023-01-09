# import pickle as pk
# import scanpy as sc
# import torch
import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset


class ModalityDataset(Dataset):
    def __init__(self, mod1, mod2):
        self.mod1 = mod1.X.toarray()
        self.mod2 = mod2.X.toarray()

    def __len__(self):
        return self.mod1.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mod1 = torch.tensor(self.mod1[idx]).float()
        mod2 = torch.tensor(self.mod2[idx]).float()
        return mod1, mod2


class ModalityNET(torch.nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        self.linear = torch.nn.ModuleList()

        for i in range(1, len(linear_layer)):
            self.linear.append(torch.nn.Linear(linear_layer[i-1], linear_layer[i]))

    def forward(self, x):
        for layer in self.linear:
            x = f.relu(layer(x))
        return x


def train(train_loader, val_loader, net, args, logger):
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr)

    training_loss = []
    val_loss = []
    criterion = torch.nn.MSELoss()
    trigger_times = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        print(epoch)
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_loss = 0
        for mod1, mod2 in tqdm(train_loader, position=0, leave=True):
            mod1, mod2 = mod1.cuda(), mod2.cuda()

            opt.zero_grad()
            out = net(mod1)
            loss = criterion(out, mod2)
            running_loss += loss.item() * mod1.size(0)
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss: {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        with torch.no_grad():
            for mod1, mod2 in val_loader:
                mod1, mod2 = mod1.cuda(), mod2.cuda()

                out = net(mod1)
                loss = criterion(out, mod2)
                running_loss += loss.item() * mod1.size(0)

            val_loss.append(running_loss / len(val_loader.dataset))
            logger.write(f'val loss: {val_loss[-1]}\n')
            logger.flush()

        # early stopping
        if len(val_loss) > 2 and round(val_loss[-1], 5) >= round(best_loss, 5):
            trigger_times += 1
            if trigger_times >= args.patience:
                logger.write(f'early stopping because val loss not decrease for {args.patience} epoch\n')
                logger.flush()
                break
        else:
            best_loss = val_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0
    return best_state_dict


def cal_rmse(ad_pred, ad_sol):
    tmp = ad_sol - ad_pred
    rmse = np.sqrt(tmp.power(2).mean())
    return rmse