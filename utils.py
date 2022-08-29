import torch
import dgl
import scanpy as sc
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from torch import nn
from dgl.nn.pytorch.conv import GraphConv


def embedding(mod, n_components, random_seed=0):
    # sc.pp.log1p(mod)
    # sc.pp.scale(mod)

    mod_reducer = TruncatedSVD(n_components=n_components, random_state=random_seed)
    mod_reducer.fit(mod)
    truncated_mod = mod_reducer.transform(mod)
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
                mod = self.encoder_acts[i * 2](mod)
                mod = self.encoder_norm[i * 2](mod)

                mod = self.hid_encoder[i * 2 + 1](mod) + temp
                mod = self.encoder_acts[i * 2 + 1](mod)
                mod = self.encoder_norm[i * 2 + 1](mod)

            mod = self.hid_encoder[-1](mod)
            mod = self.encoder_acts[-1](mod)
            mod = self.encoder_norm[-1](mod)
        else:
            for i in range(self.num_layer):
                mod = self.hid_encoder[i](mod)
                mod = self.encoder_acts[i](mod)
                mod = self.encoder_norm[i](mod)
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
                mod = self.decoder_acts[i * 2](mod)
                mod = self.decoder_norm[i * 2](mod)
        else:
            for i in range(self.num_layer):
                mod = self.hid_decoder[i](mod)
                mod = self.decoder_acts[i](mod)
                mod = self.decoder_norm[i](mod)
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
                mod = self.mod_acts[i * 2 + 1](mod)
                mod = self.mod_norm[i * 2 + 1](mod)

            mod = self.mod_decoder[-1](mod)
            mod = self.mod_acts[-1](mod)
        else:
            for i in range(self.num_layer - 1):
                mod = self.mod_decoder[i](mod)
                mod = self.mod_acts[i](mod)
                mod = self.mod_norm[i](mod)

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