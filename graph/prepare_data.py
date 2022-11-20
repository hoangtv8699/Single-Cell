import h5py
import numpy as np
import dgl
import torch
import pickle as pk

h5f_path = '../data/multiome/atac CD8+ T reshape.h5'
f1 = h5py.File(h5f_path, 'r')
data = f1["reshape"]

rand_graph = dgl.rand_graph(8647, 100000)
src_nodes = rand_graph.edges()[0]
dst_nodes = rand_graph.edges()[1]

graphs = []
for i in range(data.shape[0]):
    graph = dgl.graph((src_nodes, dst_nodes))
    graph.ndata['h'] = torch.tensor(data[i])
    graphs.append(graph)

pk.dump(graphs, open('../data/atac graphs.pkl', 'wb'))
