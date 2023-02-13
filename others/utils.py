# import pickle as pk
# import scanpy as sc
# import torch
from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from attention import MultiHeadAttention


class ModalityDataset(Dataset):
    def __init__(self, mod1, mod2_pred, mod2):
        self.mod1 = mod1
        self.mod2_pred = mod2_pred
        self.mod2 = mod2

    def __len__(self):
        return self.mod1.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mod1 = torch.tensor(self.mod1[idx]).float()
        mod2_pred = torch.tensor(self.mod2_pred[idx]).float()
        mod2 = torch.tensor(self.mod2[idx]).float()
        return mod1, mod2_pred, mod2


class ModalityNET(torch.nn.Module):
    def __init__(self, out_dims):
        super().__init__()
        self.atts = nn.MultiheadAttention(1, 1, dropout=0.3, batch_first=True)
        self.linear = nn.ModuleList()
        self.drop = nn.Dropout(0.3)
        self.norm = nn.ModuleList()

        self.linear.append(nn.Linear(out_dims, 256))
        self.linear.append(nn.Linear(256, 256))
        self.linear.append(nn.Linear(256, out_dims))

        self.norm.append(nn.LayerNorm(256))
        self.norm.append(nn.LayerNorm(256))
        self.norm.append(nn.LayerNorm(out_dims))

    def forward(self, x, y_pred):
        x = torch.unsqueeze(x, dim=-1)
        y_pred = torch.unsqueeze(y_pred, dim=-1)
        x, att_w = self.atts(y_pred, x, x)

        x = torch.flatten(x, start_dim=1)

        for layer, norm in zip(self.linear, self.norm):
            x = norm(layer(x))
            x = f.relu(x)
            x = self.drop(x)
        return x


def train_att(train_loader, val_loader, net, args, logger, time_train):
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr)

    training_loss = []
    val_loss = []
    criterion = torch.nn.MSELoss()
    trigger_times = 0
    trigger_times2 = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        print('epoch: ' + str(epoch))
        logger.write(f'epoch:  {epoch}\n')
        # training
        net.train()
        running_loss = 0
        for mod1, mod2_pred, mod2 in tqdm(train_loader, position=0, leave=True):
            mod1, mod2_pred, mod2 = mod1.cuda(), mod2_pred.cuda(), mod2.cuda()

            opt.zero_grad()
            out = net(mod1, mod2_pred)
            loss = criterion(out, mod2)
            loss.backward()
            opt.step()

            running_loss += loss.item() * mod1.size(0)

        training_loss.append(running_loss / len(train_loader.dataset))
        logger.write(f'training loss: {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        with torch.no_grad():
            for mod1, mod2_pred, mod2 in val_loader:
                mod1, mod2_pred, mod2 = mod1.cuda(), mod2_pred.cuda(), mod2.cuda()

                out = net(mod1, mod2_pred)
                loss = criterion(out, mod2)

                running_loss += loss.item() * mod1.size(0)

            val_loss.append(running_loss / len(val_loader.dataset))
            logger.write(f'val loss: {val_loss[-1]}\n')
            logger.flush()

        # early stopping
        if len(val_loss) > 2 and round(val_loss[-1], 5) >= round(best_loss, 5):
            trigger_times += 1
            trigger_times2 += 1
            if trigger_times >= args.patience:
                logger.write(f'early stopping because val loss not decrease for {args.patience} epoch\n')
                logger.flush()
                break
            # if trigger_times2 >= (args.patience / 2):
            #     trigger_times2 = 0
            #     args.lr = args.lr * args.weight_decay
            #     for g in opt.param_groups:
            #         g['lr'] = args.lr
            #     logger.write(f'lr reduce to {args.lr} because val loss not decrease for {args.patience / 2} epoch\n')
            #     logger.flush()

        else:
            torch.save(net, f'{args.save_model_path}{time_train}/model.pkl')
            trigger_times = 0
    return best_state_dict


def cal_rmse(ad_pred, ad_sol):
    tmp = ad_sol - ad_pred
    rmse = np.sqrt(tmp.power(2).mean())
    return rmse
