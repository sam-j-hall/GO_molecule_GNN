import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layers=4, emb_dim=100, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gin', heads=None, drop_ratio=0.5, graph_pooling='sum'):
        '''
        Text
        '''
        super(GNN, self).__init__()

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gnn_type = gnn_type
        self.heads = heads
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        self._initialize_weights()

        # --- Sanity check the number of layers
        if self.num_layers < 2:
            raise ValueError('Number of GNN layers mustye be greater than 1')
        
        # --- Pooling functions
        if self.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif self.graph_pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Invalid graph pooling type')
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i, in_c, out_c in zip(range(num_layers), in_channels, out_channels):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(in_c, out_c))
                self.batch_norms.append(torch.nn.BatchNorm1d(out_c))
            else:
                ValueError('Undefined GNN type called')

        self.mlp = torch.nn.Linear(self.out_channels[-1], self.num_tasks)

    def forward(self, batched_data):
        
        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        batch = batched_data.batch
        graph_indices = (batched_data.atom_num).unsqueeze(1)
        graph_sizes = torch.bincount(batch)

        h_list = [x]

        for layer in range(self.num_layers):

            x = self.convs[layer](x, edge_index)
            x = self.batch_norms[layer](x)

            if layer == self.num_layers - 1:
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            h_list.append(x)

        node_representation = h_list[-1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        zero_tensor = torch.tensor([0], device=device)

        graph_sizes_list = graph_sizes.cpu().tolist()
        graph_indices_list = graph_indices.cpu().tolist()

        cumulative_sizes = [sum(graph_sizes_list[:i]) for i in range(len(graph_sizes_list))]
        modified_indices = [graph_indices_list[i][0] + cumulative_sizes[i] for i in range(len(graph_indices_list))]

        node_select = node_representation[modified_indices]

        h_graph = self.pool(node_representation, batched_data.batch)

        w1 = 0.5
        h_weight = w1 * h_graph
        h_new = h_weight + node_select

        p = torch.nn.LeakyReLU(0.1)
        out = p(self.mlp(h_new))

        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

class GNN_node(torch.nn.Module):

    def __init__(self, num_tasks, num_layers=4, emb_dim=100, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gin', heads=None, drop_ratio=0.5, graph_pooling='sum'):
        '''
        Text
        '''
        super(GNN_node, self).__init__()

        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gnn_type = gnn_type
        self.heads = heads
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        self._initialize_weights()

        # --- Sanity check the number of layers
        if self.num_layers < 2:
            raise ValueError('Number of GNN layers mustye be greater than 1')
        
        self.convs = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()
        self.lin = torch.nn.ModuleList()

        for i, in_c, out_c in zip(range(num_layers), in_channels, out_channels):
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(in_c, out_c))
                self.lin.append(torch.nn.Linear(in_c, out_c))
                #self.batch_norms.append(torch.nn.BatchNorm1d(out_c))
            else:
                ValueError('Undefined GNN type called')

    def forward(self, batched_data):

        x = batched_data.x.float()
        edge_index = batched_data.edge_index

        for layer in range(self.num_layers):

            x = F.relu(self.convs[layer](x, edge_index) + self.lin[layer](x))

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)