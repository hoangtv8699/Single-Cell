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
    def __init__(self, mod1, mod2):
        self.mod1 = mod1.X.toarray()
        self.mod2 = mod2.X.toarray()

    def __len__(self):
        return self.mod1.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mod1 = torch.tensor(self.mod1[idx]).int()
        mod2 = torch.tensor(self.mod2[idx]).float()
        return mod1, mod2


class ModalityDataset2(Dataset):
    def __init__(self, mod1, mod1_domain, mod2):
        self.mod1 = mod1.X.toarray()
        self.mod1_domain = mod1_domain.X.toarray()
        self.mod2 = mod2.X.toarray()

    def __len__(self):
        return self.mod1.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mod1 = torch.tensor(self.mod1[idx]).int()
        mod1_domain = torch.tensor(self.mod1_domain[idx]).int()
        mod2 = torch.tensor(self.mod2[idx]).float()
        return mod1, mod1_domain, mod2


class ModalityNET(torch.nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.embed = nn.Embedding(2, 8)
        self.atts1 = nn.MultiheadAttention(8, 4, dropout=0.3)
        self.atts2 = nn.MultiheadAttention(8, 4, dropout=0.3)
        self.linear = nn.ModuleList()
        self.drop = nn.Dropout(0.3)

        self.linear.append(nn.Linear(in_dims * 8, 256))
        self.linear.append(nn.Linear(256, 256))
        # self.linear.append(nn.Linear(256, 256))
        # self.linear.append(nn.Linear(256, 256))
        self.linear.append(nn.Linear(256, out_dims))


    def forward(self, x):
        x = self.embed(x)
        x, att_w = self.atts1(x, x, x)
        x, att_w = self.atts2(x, x, x)

        x = torch.flatten(x, start_dim=1)

        for layer in self.linear:
            x = f.relu(layer(x))
            x = self.drop(x)
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(self, in_dims, in_domain_dims, out_dims):
        super().__init__()
        self.embed = nn.Embedding(2, 4)
        self.atts1 = MultiHeadAttention(4, 4, 4, 4, 1)
        self.atts2 = MultiHeadAttention(4, 4, 4, 4, 1)
        self.linear = nn.ModuleList()
        self.drop = nn.Dropout(0.3)
        self.norm = nn.ModuleList()

        self.linear.append(nn.Linear(60 * 4, 256))
        self.linear.append(nn.Linear(256, 256))

        self.norm.append(nn.LayerNorm(256))
        self.norm.append(nn.LayerNorm(256))

        # self.q = nn.Linear(in_domain_dims * 4, out_dims)
        self.out = nn.Linear(256, out_dims)


    def forward(self, x, x_domain, return_attention=False):
        # embed x
        x = self.embed(x)
        x_domain = self.embed(x_domain)
        # make query
        # x_domain = torch.flatten(x_domain, start_dim=1)
        # q = self.q(x_domain)
        # q = torch.unsqueeze(q, -1)
        # attention
        x, att_w1 = self.atts1(x_domain, x, x, return_attention=True)
        # x = self.drop(x)
        # x, att_w2 = self.atts2(x, x, x, return_attention=True)
        # x = self.drop(x)
        # linear
        x = torch.flatten(x, start_dim=1)
        for linear, norm in zip(self.linear, self.norm):
            x = norm(linear(x))
            x = f.relu(x)
            x = self.drop(x)

        x = self.out(x)
        x = f.relu(x)
        # x = self.drop(x)

        if return_attention:
            return x, att_w1
        else:
            return x

class Discriminator(torch.nn.Module):
    def __init__(self, out_dims):
        super().__init__()
        self.linear = nn.ModuleList()
        self.drop = nn.Dropout(0.1)

        self.linear.append(nn.Linear(out_dims, 256))
        self.out = nn.Linear(256, 1)


    def forward(self, x):
        for linear in self.linear:
            x = f.relu(linear(x))
            x = self.drop(x)

        x = self.out(x)
        x = torch.sigmoid(x)
        return x


class ModalityNET2(torch.nn.Module):
    def __init__(self, in_dims, in_domain_dims, out_dims):
        super().__init__()
        self.AE = AutoEncoder(in_dims, in_domain_dims, out_dims)
        self.D = Discriminator(out_dims)


def train_att(train_loader, val_loader, test_loader, net, args, logger):
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
        print(epoch)
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_loss = 0
        for mod1, mod1_domain, mod2 in tqdm(train_loader, position=0, leave=True):
            mod1, mod1_domain, mod2 = mod1.cuda(), mod1_domain.cuda(), mod2.cuda()

            opt.zero_grad()
            out = net(mod1, mod1_domain)
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
            for mod1, mod1_domain, mod2 in val_loader:
                mod1, mod1_domain, mod2 = mod1.cuda(), mod1_domain.cuda(), mod2.cuda()

                out = net(mod1, mod1_domain)
                loss = criterion(out, mod2)

                running_loss += loss.item() * mod1.size(0)


            val_loss.append(running_loss / len(val_loader.dataset))
            logger.write(f'val loss: {val_loss[-1]}\n')
            logger.flush()

        # testing
        net.eval()
        with torch.no_grad():
            outs = []
            labels = []
            for mod1, mod1_domain, mod2 in test_loader:
                mod1, mod1_domain, mod2 = mod1.cuda(), mod1_domain.cuda(), mod2.cuda()

                out = net(mod1, mod1_domain)

                outs.append(out.detach().cpu().numpy())
                labels.append(mod2.detach().cpu().numpy())

            outs = np.concatenate(outs, axis=0)
            labels = np.concatenate(labels, axis=0)
            rmse = cal_rmse(csr_matrix(outs), csr_matrix(labels))
            logger.write(f'rmse on test set: {rmse}\n')
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
            best_loss = val_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0
            trigger_times2 = 0
    return best_state_dict


def train(train_loader, val_loader, test_loader, net, args, logger):
    net.cuda()
    opt_D = torch.optim.Adam(net.parameters(), args.lr)
    opt_AE = torch.optim.Adam(net.AE.parameters(), args.lr)

    AE_training_loss = []
    D_training_loss = []
    AE_val_loss = []
    D_val_loss = []
    criterion_AE = torch.nn.MSELoss()
    criterion_D = torch.nn.BCELoss()
    trigger_times = 0
    trigger_times2 = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in range(args.epochs):
        print(epoch)
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        D_running_loss = 0
        AE_running_loss = 0
        for mod1, mod1_domain, mod2 in tqdm(train_loader, position=0, leave=True):
            mod1, mod1_domain, mod2 = mod1.cuda(), mod1_domain.cuda(), mod2.cuda()

            opt_D.zero_grad()
            D_real_out = net.D(mod2)
            D_loss_real = criterion_D(D_real_out, torch.ones(D_real_out.shape[0], 1).cuda())
            D_loss_real.backward()
            opt_D.step()

            opt_D.zero_grad()
            out_mod2 = net.AE(mod1, mod1_domain)
            D_fake_out = net.D(out_mod2)
            D_loss_fake = criterion_D(D_fake_out, torch.zeros(D_fake_out.shape[0], 1).cuda())
            D_loss_fake.backward()
            opt_D.step()

            D_running_loss += torch.add(D_loss_real, D_loss_fake) * 0.5 * mod1.size(0)

            opt_D.zero_grad()
            out = net.AE(mod1, mod1_domain)
            AE_loss = criterion_AE(out, mod2)
            AE_running_loss += AE_loss.item() * mod1.size(0)
            AE_loss.backward()
            opt_AE.step()

        AE_training_loss.append(AE_running_loss / len(train_loader.dataset))
        D_training_loss.append(D_running_loss / len(train_loader.dataset))
        logger.write(f'AE training loss: {AE_training_loss[-1]}\n')
        logger.write(f'D training loss: {D_training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        D_running_loss = 0
        AE_running_loss = 0
        with torch.no_grad():
            for mod1, mod1_domain, mod2 in val_loader:
                mod1, mod1_domain, mod2 = mod1.cuda(), mod1_domain.cuda(), mod2.cuda()

                D_real_out = net.D(mod2)
                D_loss_real = criterion_D(D_real_out, torch.ones(D_real_out.shape[0], 1).cuda())

                out_mod2 = net.AE(mod1, mod1_domain)
                D_fake_out = net.D(out_mod2)
                D_loss_fake = criterion_D(D_fake_out, torch.zeros(D_fake_out.shape[0], 1).cuda())

                D_running_loss += torch.add(D_loss_real, D_loss_fake) * 0.5 * mod1.size(0)

                out = net.AE(mod1, mod1_domain)
                AE_loss = criterion_AE(out, mod2)
                AE_running_loss += AE_loss.item() * mod1.size(0)

            AE_val_loss.append(AE_running_loss / len(val_loader.dataset))
            D_val_loss.append(D_running_loss / len(val_loader.dataset))
            logger.write(f'AE val loss: {AE_val_loss[-1]}\n')
            logger.write(f'D val loss: {D_val_loss[-1]}\n')
            logger.flush()

        # testing
        net.eval()
        with torch.no_grad():
            outs = []
            labels = []
            for mod1, mod1_domain, mod2 in test_loader:
                mod1, mod1_domain, mod2 = mod1.cuda(), mod1_domain.cuda(), mod2.cuda()

                out = net.AE(mod1, mod1_domain)
                if len(outs) == 0:
                    outs = out.detach().cpu().numpy()
                    labels = mod2.detach().cpu().numpy()
                else:
                    outs = np.concatenate((outs, out.detach().cpu().numpy()), axis=0)
                    labels = np.concatenate((labels, mod2.detach().cpu().numpy()), axis=0)
            rmse = cal_rmse(csr_matrix(outs), csr_matrix(labels))
            logger.write(f'rmse on test set: {rmse}\n')
            logger.flush()

        # early stopping
        if len(AE_val_loss) > 2 and round(AE_val_loss[-1], 5) >= round(best_loss, 5):
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
            best_loss = AE_val_loss[-1]
            best_state_dict = net.state_dict()
            trigger_times = 0
            trigger_times2 = 0
    return best_state_dict


def cal_rmse(ad_pred, ad_sol):
    tmp = ad_sol - ad_pred
    rmse = np.sqrt(tmp.power(2).mean())
    return rmse