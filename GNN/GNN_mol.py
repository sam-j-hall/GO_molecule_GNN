import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GCNConv,GINConv,GINEConv,GATv2Conv,MLP,SAGEConv
import torch.nn.functional as F
from torch.nn import ModuleList, Dropout, Linear

from GraphNet import *

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=4, emb_dim=100, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gin', heads=None, drop_ratio=0.5, graph_pooling="sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self._initialize_weights()

        ## Sanity check number of layers
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ## Defiie message passing function
        self.gnn_node = GNN_node(num_layer, emb_dim ,in_channels, out_channels, drop_ratio=drop_ratio, gnn_type=gnn_type, heads=heads)
        
        ## Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        ## Define final linear ML layer
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.out_channels[-1], self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.out_channels[-1], self.num_tasks)      	


    def forward(self, batched_data):

        node_embedding = self.gnn_node(batched_data)

        graph_embedding = self.pool(node_embedding, batched_data.batch)

        p = torch.nn.LeakyReLU(0.1)
        out = p(self.graph_pred_linear(graph_embedding))

        return out#, node_embedding
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


## GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, in_channels, out_channels, drop_ratio=0.5, gnn_type='gin', heads=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layer = num_layer
        self.heads = heads

        ## Sanity check number of layers
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        ## Set GNN layers based on type chosen
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i, in_c, out_c in zip(range(num_layer), in_channels, out_channels):
            if gnn_type == 'gin':
                mlp = MLP([in_c, in_c, out_c])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
            if gnn_type == 'gine':
                mlp = MLP([in_c, in_c, out_c])
                self.convs.append(GINEConv(nn=mlp, train_eps=False, edge_dim=5))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_c, out_c))
            elif gnn_type == 'gat':
                if i == 1:
                    self.convs.append(GATv2Conv(int(in_c), out_c, heads=int(heads), edge_dim=5))
                else:
                    self.convs.append(GATv2Conv(int(in_c*heads), out_c, heads=int(heads), edge_dim=5))
            elif gnn_type == 'sage':
                self.convs.append(SAGEConv(in_c, out_c))
            # elif gnn_type='mpnn':
            #     nn = Sequential(Linear(in_c, in_c), ReLU(), Linear(in_c, out_c * out_c))
            #     self.convs.append (NNConv(in_c, in_c, nn))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(out_c))


    def forward(self, batched_data):

        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr.float()
        batch = batched_data.batch
   
        #edge_weight = None

        ## computing input node embedding
        h_list = [x]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index)#, edge_attr=edge_attr)

            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)
             
        node_representation = h_list[-1]

        return node_representation

class GraphNet(torch.nn.Module):
    '''
    '''
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_channels: int,
                 out_channels: int,
                 gat_hidd: int,
                 gat_out: int,
                 n_layers: int=3,
                 n_targets: int=200):
        '''
        '''
        super().__init__()
        assert n_layers > 0

        # --- prepare parameters
        feat_in_node = node_dim + 2*edge_dim + gat_out
        feat_in_edge = 2*out_channels + edge_dim + gat_out
        feat_in_glob = 2*out_channels + gat_out
        node_model_params0 = {'feat_in': feat_in_node,
                             'feat_hidd': hidden_channels,
                             'feat_out': out_channels}
        edge_model_params0 = {'feat_in': feat_in_edge,
                             'feat_hidd': hidden_channels,
                             'feat_out': out_channels}
        global_model_params0 = {'feat_in': feat_in_glob,
                               'feat_hidd': hidden_channels,
                               'feat_out': out_channels}

        all_params = {'graphnet0': {'node_model_params': node_model_params0,
                                   'edge_model_params': edge_model_params0,
                                   'global_model_params': global_model_params0,
                                   'gat_in': node_dim,
                                   'gat_hidd': gat_hidd,
                                   'gat_out': gat_out}}

        for i in range(1, n_layers):
            all_params[f'graphnet{i}'] = {
                'node_model_params': {'feat_in': 4*out_channels,
                                      'feat_hidd': hidden_channels,
                                      'feat_out': out_channels},
                'edge_model_params': {'feat_in': 4*out_channels,
                                      'feat_hidd': hidden_channels,
                                      'feat_out': out_channels},
                'global_model_params': {'feat_in': 3*out_channels,
                                        'feat_hidd': hidden_channels,
                                        'feat_out': out_channels},
                'gat_in': node_dim,
                'gat_hidd': gat_hidd,
                'gat_out': gat_out
            }   

            graphnets = []
            for v in all_params.values():
                graphnets.append(GraphNetwork(**v))

            self.graphnets = ModuleList(graphnets)
            self.dropout = Dropout(p=0.5)
            self.output_dense = Linear(out_channels, n_targets)
            self.reset_parameters()

    def reset_parameters(self):
        tensor = torch.nn.init.orthogonal_(self.output_dense.weight.data)

        if len(tensor.shape) == 3:
            fan_in = tensor.shape[-1].numul()
        else:
            fan_in = tensor.shape[1]

        with torch.no_grad():
            if len(tensor.shape) == 3:
                axis = [0,1]
            else:
                axis = 1
            var, mean = torch.var_mean(tensor, dim=axis, keepdim=True)
            tensor = (tensor - mean) / (var) ** 0.5
                
            tensor *= (1/fan_in)**0.5
        return tensor

    def forward(self, graph: Any) -> torch.Tensor:
        for graphnet in self.graphnets:
            graph = graphnet(graph)

        x = global_mean_pool(graph.x, graph.batch)

        x = self.dropout(x)
        out = self.output_dense(x)

        return out