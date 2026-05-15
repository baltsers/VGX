import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f

class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        #self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, batch, cuda=False, device=None):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda, device=device)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        return h_i
        #embeddings = h_i.sum(dim=1)
        #embeddings.append(h_i.sum(dim=1).tolist())
        #ggnn_sum = self.classifier(h_i.sum(dim=1))
        #result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        #del h_i
        #return embeddings
