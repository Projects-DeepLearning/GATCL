import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.utils import spectral_norm
from torch_geometric.nn import GATConv


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = spectral_norm(nn.Bilinear(n_h, n_h, 1))
        self.bias = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, c, h_pl, h_mi):
        c_x = c.expand_as(h_pl)
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        logits = torch.cat((sc_1, sc_2), dim=1) + self.bias
        return logits


class AttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.attention_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, emb, mask=None):
        attention_scores = self.attention_layer(emb).squeeze()
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        global_emb = torch.sum(emb * attention_weights.unsqueeze(-1), dim=1)
        return F.normalize(global_emb, p=2, dim=1)


# from torch_geometric.nn import GATConv
class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        self.conv1 = GATConv(self.in_features, 64, heads=4, dropout=self.dropout, self_loop=False)
        self.conv2 = GATConv(4 * 64, self.in_features, dropout=self.dropout, self_loop=False)

        self.reset_parameters()
        self.disc = Discriminator(256)

        self.sigm = nn.Sigmoid()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, feat, feat_a, edge_index, edge_weight):
        z = F.dropout(feat, self.dropout, self.training)
        z = self.conv1(z, edge_index, edge_attr=edge_weight)
        z = self.act(z)

        hidden_emb = z

        h = F.dropout(z, self.dropout, self.training)
        h = self.conv2(h, edge_index, edge_attr=edge_weight)

        emb = self.act(z)

        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = self.conv1(z_a, edge_index, edge_attr=edge_weight)
        z_a = self.act(z_a)

        emb_a = self.act(z_a)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        pooling = AttentionPooling(in_features=emb.shape[-1], hidden_dim=128).to(device)
        g = pooling(emb.to(device), self.graph_neigh.bool().to(device))
        g = self.sigm(g)

        pooling = AttentionPooling(in_features=emb_a.shape[-1], hidden_dim=128).to(device)
        g_a = pooling(emb_a.to(device), self.graph_neigh.bool().to(device))
        g_a = self.sigm(g_a)

        ret = self.disc(g, emb, emb_a)
        ret_a = self.disc(g_a, emb_a, emb)

        return hidden_emb, h, ret, ret_a

