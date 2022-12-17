# import pickle as pk
# import scanpy as sc
# import torch
import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.nn import SAGEConv


class GraphDataset(Dataset):
    def __init__(self, mod1, mod2):
        self.mod1 = np.expand_dims(mod1, axis=-1)
        self.mod2 = mod2

    def __len__(self):
        return self.mod1.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mod1 = torch.tensor(self.mod1[idx]).float()
        mod2 = torch.tensor(self.mod2[idx]).float()
        return mod1, mod2


class GCN(torch.nn.Module):
    def __init__(self, gat_layer, linear_layer):
        super().__init__()
        self.gat = torch.nn.ModuleList()
        self.linear = torch.nn.ModuleList()

        for i in range(1, len(gat_layer)):
            # self.gat.append(GATv2Conv(gat_layer[i - 1], gat_layer[i], edge_dim=1))
            self.gat.append(SAGEConv(gat_layer[i - 1], gat_layer[i]))
        for i in range(1, len(linear_layer)):
            self.linear.append(nn.Linear(linear_layer[i - 1], linear_layer[i]))

    # def forward(self, x, edge_index, edge_weight):
    def forward(self, x, edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        # edge_weight: Graph weight of shape [num_edges, 1]
        for layer in self.gat:
            # x, attw = layer(x, edge_index, edge_weight, return_attention_weights=True)
            x = layer(x, edge_index)
            x = f.relu(x)
        x = torch.squeeze(x)
        for layer in self.linear:
            x = f.relu(layer(x))
        return x


def train(training_set, val_set, net, args, logger, edges, weights):
    edges = torch.tensor(edges).cuda()
    weights = torch.tensor(weights).float().cuda()
    print('train GCN')
    net.cuda()
    opt = torch.optim.Adam(net.parameters(), args.lr)

    training_loss = []
    val_loss = []
    criterion = torch.nn.MSELoss()
    trigger_times = 0
    best_loss = 10000
    best_state_dict = net.state_dict()

    for epoch in tqdm(range(args.epochs), position=0, leave=True):
        logger.write(f'epoch:  {epoch}\n')

        # training
        net.train()
        running_loss = 0
        for mod1, mod2 in tqdm(training_set, position=0, leave=True):
            mod1, mod2 = mod1.cuda(), mod2.cuda()

            opt.zero_grad()
            # out = net(mod1, edges, weights)
            out = net(mod1, edges)
            loss = criterion(out, mod2)
            running_loss += loss.item()
            loss.backward()
            opt.step()

        training_loss.append(running_loss / len(training_set))
        logger.write(f'training loss: {training_loss[-1]}\n')
        logger.flush()

        # validating
        net.eval()
        running_loss = 0
        with torch.no_grad():
            for mod1, mod2 in val_set:
                mod1, mod2 = mod1.cuda(), mod2.cuda()

                # optimize encoder contrastive
                # out = net(mod1, edges, weights)
                out = net(mod1, edges)
                loss = criterion(out, mod2)
                running_loss += loss.item()

            val_loss.append(running_loss / len(val_set))
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
        print(epoch)
    return best_state_dict


def cal_rmse(ad_pred, ad_sol):
    tmp = ad_sol - ad_pred
    rmse = np.sqrt(tmp.power(2).mean())
    return rmse