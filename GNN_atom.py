import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, MLP, GINEConv, AttentionalAggregation
import torch.nn.functional as F
from icecream import ic

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

        ic('enter')
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
        ic('here')
        h_node, h_select = self.gnn_node(batched_data)

        #ic(batched_data.batch.shape)
        #ic(batched_data.atom_num)
     #   atom = batched_data.atom_num
        #ic(h_node[atom])
        #ic(h_node)

#        h_graph = h_node[atom]
        h_graph = self.pool(h_node, batched_data.batch)

        #ic(h_graph.shape)
        #ic(h_graph)

#        exit()
        # Compute the weighted sum of x_batch and x_sum
        w1 = 0.5 # Adjust this value as needed
        h_weight = w1 * h_graph

        if h_select.dim() == 1:
            h_select = h_select.unsqueeze(0)
        h_out = torch.cat((h_select, h_weight), dim=1)
        #h_out=h_select+h_weight

        #print(h_out.shape)
        #out = F.relu(self.lin1(out))
        out = torch.sigmoid(self.graph_pred_linear(h_out))

#        p = torch.nn.LeakyReLU(0.1)
   
#        out = p(self.graph_pred_linear(h_graph))

#        out = p(self.graph_pred_linear1(out))

        return  out, h_select, h_weight, h_out
    
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
                self.convs.append(GINEConv(nn=mlp, train_eps=False,edge_dim=7))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_c, out_c))
            elif gnn_type == 'gat':
                if i == 0:
                    self.convs.append(GATv2Conv(in_c, out_c, heads=int(heads),edge_dim=7))
                else:
                    self.convs.append(GATv2Conv(int(in_c * heads), out_c, heads=1,edge_dim=7))
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
        edge_attr = batched_data.edge_attrs
        batch = batched_data.batch
        graph_indices = (batched_data.atom_num).unsqueeze(1)
        edge_weight = None
        
       # edge_attr=edge_attr(dtype=torch.float)
        ic(type(edge_attr))
        ic(type(x))
        ic(type(edge_index))
        ic(type(batch))

        edge_attr = edge_attr.float()  
        #batch = batch_data.batch

        # Calculate the graph sizes
        graph_sizes = torch.bincount(batch)

        # Print the graph sizes
        #print(graph_sizes)

        ## computing input node embedding
        h_list = [x]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index)

            h = F.relu(h)
   
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            h_list.append(h)
        
        #ic(h_list[0].shape)

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
        
        #print(cumulative_sizes)

        # Compute modified indices in the batch
        modified_indices = [graph_indices_list[i][0] + cumulative_sizes[i] for i in range(len(graph_indices_list))]
       
        #print(modified_indices)

      #  print(node_representation, batch_index)
      #  print(node_select)
       
        #        print(node_representation,batch_index)
        
        node_select = node_representation[modified_indices]
        #print(node_representation,node_select)

        return node_representation, node_select