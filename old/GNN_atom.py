import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, MLP, GINEConv, AttentionalAggregation, SAGEConv, NNConv, PNAConv, CGConv, ClusterGCNConv
import torch.nn.functional as F

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

        ## Define message passing function
        self.gnn_node = GNN_node(num_layer, emb_dim ,in_channels,out_channels,drop_ratio = drop_ratio, gnn_type = gnn_type,heads=heads)
        
        ## Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            #self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(self.out_channels[-1], 2*self.out_channels[-1]), torch.nn.BatchNorm1d(2*self.out_channels[-1]), torch.nn.ReLU(), torch.nn.Linear(2*self.out_channels[-1], self.out_channels[-1])))
            self.pool = AttentionalAggregation(torch.nn.Linear(self.out_channels[-1], 1))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        ## Define final ML layer
        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.out_channels[-1], self.num_tasks)
        else:
            
#             self.graph_pred_linear=torch.nn.Sequential(
#             torch.nn.Linear(2*self.out_channels[-1], self.out_channels[-1],),
#             torch.nn.ReLU(),  # You can use other activation functions here if needed
#             torch.nn.Linear(self.out_channels[-1], self.out_channels[-1],),
#             torch.nn.ReLU(),
#             torch.nn.Linear(self.out_channels[-1], self.num_tasks))
            
            self.graph_pred_linear = torch.nn.Linear(self.out_channels[-1], self.num_tasks)
            #self.graph_pred_linear1 = torch.nn.Linear(200, 200)
	

    def forward(self, batched_data):

        # --- Start with batched date from data loader
        # ic(batched_data)

        # --- Run batched data throught the convolution layers
        h_node, h_select = self.gnn_node(batched_data)

        # --- Output gives you a tensor of shape (number of atoms in batch, number of features on final layer in gnn_node)
        # --- 
        # ic(type(h_node))
        # ic(h_node.shape)
        # ic(h_node[0])
        # ic(len(h_node[0]))
        # ic(h_select)
        # ic(len(h_select))

        # --- batched_data lists all the x feature vectors of every molecule in the batch
        # --- from the first atom x vector to the last, so length sum of number of atoms in batch
        # --- The batched_data.batch gives a tensor which is used to label all atoms in th same molecule
        # --- [0,0,0,0... for how many atoms there are, 1,1,1... and so on to, x,x,x... the number of molecules]
        # ic(batched_data.batch)
        # ic(batched_data.batch.shape)
        # ic(batched_data.atom_num)
        # atom = batched_data.atom_num
        # ic(atom)
        # ic(h_node[atom])
        # ic(h_node)

#        h_graph = h_node[atom]
        # --- This pools together all atoms in the same molecule together
        h_graph = self.pool(h_node, batched_data.batch)

        # --- Output give you tensor of shape (number of items in batch, number of features in final layer in gnn_node)
        # ic(h_graph.shape)
        # ic(h_graph)
        # ic(len(h_graph))

        # ic(h_graph[0])
        # ic(batched_data.atom_num[0])
        # ic(h_node[batched_data.atom_num[0]])
        # ic(h_select[0])

#        exit()
        # Compute the weighted sum of x_batch and x_sum
        w1 = 0.5 # Adjust this value as needed
        h_weight = w1 * h_graph
        h_new = h_weight + h_select

        #h_new = torch.cat((h_graph, h_select), dim=1)


        #if h_select.dim() == 1:
         #   h_select = h_select.unsqueeze(0)
       # ic(h_select.shape)
        #ic(h_weight.shape)
        #h_out = torch.cat((h_select, h_weight), dim=1)
        #h_out=h_select+h_weight

        #print(h_out.shape)
        #out = F.relu(self.lin1(out))

#        ic(h_out.shape)

        #out = torch.sigmoid(self.graph_pred_linear(h_new))
        #out = self.graph_pred_linear(h_new)

        p = torch.nn.LeakyReLU(0.1)
   
        out = p(self.graph_pred_linear(h_new))

#        out = p(self.graph_pred_linear1(out))

        return  out, h_select#, h_weight, h_out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)



### GNN to generate node embedding
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


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        ## Set GNN layers based on type chosen
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i, in_c, out_c in zip(range(num_layer), in_channels, out_channels):
            if gnn_type == 'gin':
                mlp = MLP([in_c, in_c, out_c])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
            elif gnn_type =='gine':
                mlp = MLP([in_c, in_c, out_c])
                self.convs.append(GINEConv(nn=mlp, train_eps=False,edge_dim=6))
            elif gnn_type == 'gcn':
                self.convs.append(SAGEConv(in_c, out_c))
            elif gnn_type == 'gat':
                if i == 0:
                    self.convs.append(GATv2Conv(in_c, out_c, heads=int(heads),edge_dim=6))
                else:
                    self.convs.append(GATv2Conv(int(in_c * heads), out_c, heads=1,edge_dim=6))
            # elif gnn_type='mpnn':
            #     nn = Sequential(Linear(in_c, in_c), ReLU(), Linear(in_c, out_c * out_c))
            #     self.convs.append (NNConv(in_c, in_c, nn))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))
            if gnn_type != 'gat':
                self.batch_norms.append(torch.nn.BatchNorm1d(out_c))
            else:
#                 if i==0:
                   
#                     self.batch_norms.append(torch.nn.BatchNorm1d(out_c))
#                 else :
                self.batch_norms.append(torch.nn.BatchNorm1d(out_c))


    def forward(self, batched_data):

        x = batched_data.x.float()
        edge_index = batched_data.edge_index
        edge_attr = batched_data.edge_attr
        batch = batched_data.batch
        graph_indices = (batched_data.atom_num).unsqueeze(1)
        edge_weight = None
        pos = batched_data.pos

        # for i in range(len(pos)):
        #     x = x + sum(pos[i])

        #edge_attr=edge_attr(dtype=torch.float)
#        ic(type(edge_attr))
 #       ic(type(x))
  #      ic(type(edge_index))
   #     ic(type(batch))

        edge_attr = edge_attr.float()  
        #batch = batch_data.batch

        # Calculate the graph sizes
        graph_sizes = torch.bincount(batch)

        # Print the graph sizes
        #print(graph_sizes)

        ## computing input node embedding
        h_list = [x]

        for layer in range(self.num_layer):

            x = self.convs[layer](x, edge_index)#, edge_attr=edge_attr)
   
            x = self.batch_norms[layer](x)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)

            h_list.append(x)

            # for layer in range(self.num_layer):            #     node_representation += h_list[layer]
                
        #considering the last layer representation        
        node_representation = h_list[-1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        zero_tensor = torch.tensor([0],device=device)
        
        #cum_graph_indices = torch.cat((zero_tensor, torch.cumsum(batch_index.flatten(), dim=0)))
        
        graph_sizes_list = graph_sizes.cpu().tolist()
        graph_indices_list = graph_indices.cpu().tolist()
        
        #print(graph_sizes_list,graph_indices_list)
        # Compute cumulative sum of graph sizes
        cumulative_sizes = [sum(graph_sizes_list[:i]) for i in range(len(graph_sizes_list))]
       # cumulative_sizes_list = cumulative_sizes.)
        

        # Compute modified indices in the batch
        modified_indices = [graph_indices_list[i][0] + cumulative_sizes[i] for i in range(len(graph_indices_list))]
        #print(modified_indices)

      #  print(node_representation, batch_index)
      #  print(node_select)
       
        #        print(node_representation,batch_index)
        
        node_select = node_representation[modified_indices]
        #print(node_representation,node_select)
        
        return node_representation, node_select
    
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
            torch.nn.Linear(32, 5 * 64)
        )

        self.conv1 = NNConv(5, 64, nn1, aggr='mean')
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

        node_select = node_rep[modified_indicies]

        out = global_mean_pool(x, batch)

        # Compute the weighted sum of x_batch and x_sum
        w1 = 0.5 # Adjust this value as needed
        h_weight = w1 * out
        h_new = h_weight + node_select

        out = self.mlp(h_new)

        return out, node_select

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


        self.conv1 = PNAConv(in_channels=5, out_channels=64, aggregators=aggregators, scalers=scalers,
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

        node_select = node_rep[modified_indicies]

        out = global_mean_pool(x, batch)

        # Compute the weighted sum of x_batch and x_sum
        w1 = 0.5 # Adjust this value as needed
        h_weight = w1 * out
        h_new = h_weight + node_select

        out = self.mlp(h_new), node_select

        return out
    
class CGC(torch.nn.Module):

    def __init__(self, num_tasks, num_layer=4, emb_dim=100, in_channels=[33,100,100,100], out_channels=[100,100,100,100],
                 gnn_type='gin', heads=None, drop_ratio=0.5, graph_pooling="sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(CGC, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.conv1 = CGConv(channels=[15, 64], dim=6, aggr='mean', batch_norm=True)

        self.conv2 = CGConv(channels=[64, 128], dim=6, aggr='mean', batch_norm=True)

        self.conv3 = CGConv(channels=[128, 256], dim=6, aggr='mean', batch_norm=True)

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

        x = F.relu(x)
        x = F.dropout(x, self.drop_ratio, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, self.drop_ratio, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        node_rep = x

        graph_sizes_list = graph_sizes.cpu().tolist()
        graph_indices_list = graph_indices.cpu().tolist()

        cumulative_sizes = [sum(graph_sizes_list[:i]) for i in range(len(graph_sizes_list))]

        modified_indicies = [graph_indices_list[i][0] + cumulative_sizes[i] for i in range(len(graph_indices_list))]

        node_select = node_rep[modified_indicies]

        out = global_mean_pool(x, batch)

        # Compute the weighted sum of x_batch and x_sum
        w1 = 0.5 # Adjust this value as needed
        h_weight = w1 * out
        h_new = h_weight + node_select

        out = self.mlp(h_new)

        return out
    