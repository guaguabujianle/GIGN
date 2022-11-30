# %%
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import edge_softmax
import dgl

# taken from https://github.com/zjujdj/InteractionGraphNet/tree/master

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        # return torch.sigmoid(h)
        return h

class DTIConvGraph3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DTIConvGraph3, self).__init__()
    # the MPL for update the edge state
        self.mpl = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU(),
                                 nn.Linear(out_dim, out_dim),
                                 nn.LeakyReLU())

    def EdgeUpdate(self, edges):
        return {'e': self.mpl(torch.cat([edges.data['e'], edges.data['m']], dim=1))}

    def forward(self, bg, atom_feats, bond_feats):
        bg.ndata['h'] = atom_feats
        bg.edata['e'] = bond_feats
        with bg.local_scope():
            bg.apply_edges(dgl.function.u_add_v('h', 'h', 'm'))
            bg.apply_edges(self.EdgeUpdate)
            return bg.edata['e']


class DTIConvGraph3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):  # in_dim = graph module1 output dim + 1
        super(DTIConvGraph3Layer, self).__init__()
        # the MPL for update the edge state
        self.grah_conv = DTIConvGraph3(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn_layer = nn.BatchNorm1d(out_dim)

    def forward(self, bg, atom_feats, bond_feats):
        new_feats = self.grah_conv(bg, atom_feats, bond_feats)
        return self.bn_layer(self.dropout(new_feats))


class EdgeWeightAndSum(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Tanh()
        )

    def forward(self, g, edge_feats):
        with g.local_scope():
            g.edata['e'] = edge_feats
            g.edata['w'] = self.atom_weighting(g.edata['e'])
            # weights = g.edata['w']  # temporary version
            h_g_sum = dgl.sum_edges(g, 'e', 'w')
        return h_g_sum  # normal version
        # return h_g_sum, weights  # temporary version


class EdgeWeightedSumAndMax(nn.Module):
    """
    for normal use, please delete the 'temporary version' line and meanwhile recover the 'normal version'
    """
    def __init__(self, in_feats):
        super(EdgeWeightedSumAndMax, self).__init__()
        self.weight_and_sum = EdgeWeightAndSum(in_feats)

    def forward(self, bg, edge_feats):
        h_g_sum = self.weight_and_sum(bg, edge_feats)  # normal version
        # h_g_sum, weights = self.weight_and_sum(bg, edge_feats)  # temporary version
        with bg.local_scope():
            bg.edata['e'] = edge_feats
            h_g_max = dgl.max_edges(bg, 'e')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g  # normal version
        # return h_g, weights  # temporary version


class AttentiveGRU1(nn.Module):

    def __init__(self, node_feat_size, edge_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU1, self).__init__()

        self.edge_transform = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(edge_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, edge_feats, node_feats):
        g = g.local_var()
        g.edata['e'] = edge_softmax(g, edge_logits) * self.edge_transform(edge_feats)
        g.update_all(fn.copy_edge('e', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class AttentiveGRU2(nn.Module):
    def __init__(self, node_feat_size, edge_hidden_size, dropout):
        super(AttentiveGRU2, self).__init__()

        self.project_node = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(node_feat_size, edge_hidden_size)
        )
        self.gru = nn.GRUCell(edge_hidden_size, node_feat_size)

    def forward(self, g, edge_logits, node_feats):
        g = g.local_var()
        g.edata['a'] = edge_softmax(g, edge_logits)
        g.ndata['hv'] = self.project_node(node_feats)

        g.update_all(fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'c'))
        context = F.elu(g.ndata['c'])
        return F.relu(self.gru(context, node_feats))


class GetContext(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, graph_feat_size, dropout):
        super(GetContext, self).__init__()

        self.project_node = nn.Sequential(
            nn.Linear(node_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge1 = nn.Sequential(
            nn.Linear(node_feat_size + edge_feat_size, graph_feat_size),
            nn.LeakyReLU()
        )
        self.project_edge2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * graph_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU1(graph_feat_size, graph_feat_size,
                                           graph_feat_size, dropout)

    def apply_edges1(self, edges):
        return {'he1': torch.cat([edges.src['hv'], edges.data['he']], dim=1)}

    def apply_edges2(self, edges):
        return {'he2': torch.cat([edges.dst['hv_new'], edges.data['he1']], dim=1)}

    def forward(self, g, node_feats, edge_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.ndata['hv_new'] = self.project_node(node_feats)
        g.edata['he'] = edge_feats

        # h_u || h_uv
        g.apply_edges(self.apply_edges1)
        g.edata['he1'] = self.project_edge1(g.edata['he1'])
        # h_v || h_uv
        g.apply_edges(self.apply_edges2)
        logits = self.project_edge2(g.edata['he2'])

        return self.attentive_gru(g, logits, g.edata['he1'], g.ndata['hv_new'])


class GNNLayer(nn.Module):
    def __init__(self, node_feat_size, graph_feat_size, dropout):
        super(GNNLayer, self).__init__()

        self.project_edge = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * node_feat_size, 1),
            nn.LeakyReLU()
        )
        self.attentive_gru = AttentiveGRU2(node_feat_size, graph_feat_size, dropout)
        self.bn_layer = nn.BatchNorm1d(graph_feat_size)

    def apply_edges(self, edges):
        return {'he': torch.cat([edges.dst['hv'], edges.src['hv']], dim=1)}

    def forward(self, g, node_feats):
        g = g.local_var()
        g.ndata['hv'] = node_feats
        g.apply_edges(self.apply_edges)
        logits = self.project_edge(g.edata['he'])

        return self.bn_layer(self.attentive_gru(g, logits, node_feats))


class ModifiedAttentiveFPGNNV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPGNNV2, self).__init__()

        self.init_context = GetContext(node_feat_size, edge_feat_size, graph_feat_size, dropout)
        self.gnn_layers = nn.ModuleList()
        self.sum_node_feats = 0
        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(graph_feat_size, graph_feat_size, dropout))

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.init_context(g, node_feats, edge_feats)
        self.sum_node_feats = node_feats
        for gnn in self.gnn_layers:
            node_feats = gnn(g, node_feats)
            self.sum_node_feats = self.sum_node_feats + node_feats
        return self.sum_node_feats


class ModifiedAttentiveFPPredictorV2(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=2,
                 graph_feat_size=200,
                 dropout=0.):
        super(ModifiedAttentiveFPPredictorV2, self).__init__()

        self.gnn = ModifiedAttentiveFPGNNV2(node_feat_size=node_feat_size,
                                            edge_feat_size=edge_feat_size,
                                            num_layers=num_layers,
                                            graph_feat_size=graph_feat_size,
                                            dropout=dropout)
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(graph_feat_size, 1)
        )

    def forward(self, g, node_feats, edge_feats):
        sum_node_feats = self.gnn(g, node_feats, edge_feats)
        return sum_node_feats


class DTIPredictorV4_V2(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, num_layers, graph_feat_size, outdim_g3,
                 d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(DTIPredictorV4_V2, self).__init__()
        # graph layers for ligand and protein
        self.cov_graph = ModifiedAttentiveFPPredictorV2(node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout)

        # graph layers for ligand and protein interaction
        self.noncov_graph = DTIConvGraph3Layer(graph_feat_size+1, outdim_g3, dropout)

        # MLP predictor
        self.FC = FC(outdim_g3*2, d_FC_layer, n_FC_layer, dropout, n_tasks)
        # self.FC = nn.Linear(outdim_g3*2, 1)
        # read out
        self.readout = EdgeWeightedSumAndMax(outdim_g3)

    def forward(self, bg, bg3):
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')
        atom_feats = self.cov_graph(bg, atom_feats, bond_feats)
        bond_feats3 = bg3.edata['e']
        bond_feats3 = self.noncov_graph(bg3, atom_feats, bond_feats3)
        readouts = self.readout(bg3, bond_feats3)
        return self.FC(readouts)



# %%

if __name__ == '__main__':
    import os
    import pandas as pd
    from thop import profile
    from torch.utils.data import DataLoader
    from dataset_IGN import collate_fn, GraphDataset

    data_root = './data'
    graph_type = 'Graph_IGN'
    test2013_dir = os.path.join(data_root, 'test2013')
    test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
    test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
    test2013_loader = DataLoader(test2013_set, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=8)

    data = next(iter(test2013_loader))
    bg, bg_lp, label = data
    model =  DTIPredictorV4_V2(node_feat_size=40, edge_feat_size=21, num_layers=3,
                                            graph_feat_size=256, outdim_g3=200,
                                            d_FC_layer=200, n_FC_layer=2, dropout=0.1, n_tasks=1)
    flops, params = profile(model, inputs=(bg, bg_lp, ))
    print("params: ", params/1e6)
    print("flops: ", flops/1e9) 
# %%