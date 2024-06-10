import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv, NNConv, PNAConv, TransformerConv, SAGEConv

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
            self.batch_norms.append(torch.nn.BatchNorm1d(out_c))
            if self.gnn_type == 'gcn':
                self.convs.append(GCNConv(in_c, out_c))
            if self.gnn_type == 'sage':
                self.convs.append(SAGEConv(in_c, out_c))
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

        node_embedding = node_representation[modified_indices]

        graph_embedding = self.pool(node_representation, batched_data.batch)

        w1 = 0.0
        h_weight = w1 * graph_embedding
        h_new = h_weight + node_embedding

        p = torch.nn.LeakyReLU(0.1)
        out = p(self.mlp(h_new))

        return out, node_embedding
    
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

        for layer in range(self.num_layers -1):

            x = F.relu(self.convs[layer](x, edge_index) + self.lin[layer](x))
            #x = self.batch_norms[layer](x)
        x = self.convs[-1](x, edge_index) + self.lin[-1](x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

class nnconv(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=4, emb_dim=100, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gin', heads=None, drop_ratio=0.5, graph_pooling="sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(nnconv, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        nn1 = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 15 * 64)
        )

        self.conv1 = NNConv(15, 64, nn1, aggr='mean')
        self.batchnorm1 = torch.nn.BatchNorm1d(64)

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64 * 128)
        )

        self.conv2 = NNConv(64, 128, nn2, aggr='mean')
        self.batchnorm2 = torch.nn.BatchNorm1d(128)

        nn3 = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 128 * 256)
        )

        self.conv3 = NNConv(128, 256, nn3, aggr='mean')
        self.batchnorm3 = torch.nn.BatchNorm1d(256)

        self.mlp = torch.nn.Linear(256, 200)

    def forward(self, batched_data):

        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr.float()
        batch = batched_data.batch

         # Calculate the graph sizes
        graph_sizes = torch.bincount(batch)
        graph_indices = (batched_data.atom_num).unsqueeze(1)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batchnorm1(x)

        x = F.relu(x)
        x = F.dropout(x, self.drop_ratio, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_ratio, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batchnorm3(x)
        x = F.relu(x)

        node_rep = x

        graph_sizes_list = graph_sizes.cpu().tolist()
        graph_indices_list = graph_indices.cpu().tolist()

        cumulative_sizes = [sum(graph_sizes_list[:i]) for i in range(len(graph_sizes_list))]

        modified_indicies = [graph_indices_list[i][0] + cumulative_sizes[i] for i in range(len(graph_indices_list))]

        node_embedding = node_rep[modified_indicies]

        graph_embedding = global_mean_pool(x, batch)

        # Compute the weighted sum of x_batch and x_sum
        w1 = 0.5 # Adjust this value as needed
        h_weight = w1 * graph_embedding
        h_new = h_weight + node_embedding

        out = self.mlp(h_new)

        return out, node_embedding
    
class PNA(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=4, emb_dim=100, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gin', heads=None, drop_ratio=0.5, graph_pooling="sum", deg=None):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(PNA, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.deg = deg

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']


        self.conv1 = PNAConv(in_channels=15, out_channels=64, aggregators=aggregators, scalers=scalers,
                             deg=self.deg, edge_dim=3)
        self.batchnorm1 = torch.nn.BatchNorm1d(64)

        self.conv2 = PNAConv(in_channels=64, out_channels=128, aggregators=aggregators, scalers=scalers,
                            deg=self.deg, edge_dim=3)
        self.batchnorm2 = torch.nn.BatchNorm1d(128)

        self.conv3 = PNAConv(in_channels=128, out_channels=256, aggregators=aggregators, scalers=scalers,
                            deg=self.deg, edge_dim=3)
        self.batchnorm3 = torch.nn.BatchNorm1d(256)

        self.mlp = torch.nn.Linear(256, 200)

    def forward(self, batched_data):

        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr.float()
        batch = batched_data.batch

         # Calculate the graph sizes
        graph_sizes = torch.bincount(batch)
        graph_indices = (batched_data.atom_num).unsqueeze(1)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.batchnorm1(x)

        x = F.relu(x)
        x = F.dropout(x, self.drop_ratio, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_ratio, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.batchnorm3(x)
        x = F.relu(x)

        node_rep = x

        graph_sizes_list = graph_sizes.cpu().tolist()
        graph_indices_list = graph_indices.cpu().tolist()

        cumulative_sizes = [sum(graph_sizes_list[:i]) for i in range(len(graph_sizes_list))]

        modified_indicies = [graph_indices_list[i][0] + cumulative_sizes[i] for i in range(len(graph_indices_list))]

        node_embedding = node_rep[modified_indicies]

        graph_embedding = global_mean_pool(x, batch)

        # Compute the weighted sum of x_batch and x_sum
        w1 = 0.5 # Adjust this value as needed
        h_weight = w1 * graph_embedding
        h_new = h_weight + node_embedding

        out = self.mlp(h_new), node_embedding

        return out
    
class UniMP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, beta=True, heads=1):
        """
        Params:
        - input_dim: The dimension of input features for each node.
        - hidden_dim: The size of the hidden layers.
        - output_dim: The dimension of the output features (often equal to the
            number of classes in a classification task).
        - num_layers: The number of layer blocks in the model.
        - dropout: The dropout rate for regularization. It is used to prevent
            overfitting, helping the learning process remains generalized.
        - beta: A boolean parameter indicating whether to use a gated residual
            connection (based on equations 5 and 6 from the UniMP paper). The
            gated residual connection (controlled by the beta parameter) helps
            preventing overfitting by allowing the model to balance between new
            and existing node features across layers.
        - heads: The number of heads in the multi-head attention mechanism.
        """
        super(UniMP, self).__init__()
  
        # The list of transormer conv layers for the each layer block.
        self.num_layers = num_layers
        conv_layers = [TransformerConv(input_dim, hidden_dim//heads, heads=heads, beta=beta)]
        conv_layers += [TransformerConv(hidden_dim, hidden_dim//heads, heads=heads, beta=beta) for _ in range(num_layers - 2)]
        # In the last layer, we will employ averaging for multi-head output by
        # setting concat to True.
        conv_layers.append(TransformerConv(hidden_dim, output_dim, heads=heads, beta=beta, concat=False))
        self.convs = torch.nn.ModuleList(conv_layers)
  
        # The list of layerNorm for each layer block.
        norm_layers = [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        self.norms = torch.nn.ModuleList(norm_layers)
  
        # Probability of an element getting zeroed.
        self.dropout = dropout

    def reset_parameters(self):
        """
        Resets the parameters of the convolutional and normalization layers,
        ensuring they are re-initialized when needed.
        """
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, batched_data):
        """
        The input features are passed sequentially through the transformer
        convolutional layers. After each convolutional layer (except the last),
        the following operations are applied:
        - Layer normalization (`LayerNorm`).
        - ReLU activation function.
        - Dropout for regularization.
        The final layer is processed without layer normalization and ReLU
        to average the multi-head results for the expected output.

        Params:
        - x: node features x
        - edge_index: edge indices.

        """
        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        for i in range(self.num_layers - 1):
        # Construct the network as shown in the model architecture.
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            # By setting training to self.training, we will only apply dropout
            # during model training.
            x = F.dropout(x, p = self.dropout, training = self.training)

        # Last layer, average multi-head output.
        x = self.convs[-1](x, edge_index)

        return x