import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GCNConv,GINConv,GATv2Conv,MLP
import torch.nn.functional as F

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 4, emb_dim = 100, in_channels=[33,100,100,100],out_channels=[100,100,100,100],
                    gnn_type= 'gin', heads=None,drop_ratio = 0.5, graph_pooling = "sum"):
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
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.heads=heads
        
        self._initialize_weights()


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

       
        self.gnn_node = GNN_node(num_layer, emb_dim ,in_channels,out_channels,drop_ratio = drop_ratio, gnn_type = gnn_type,heads=heads)
        

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.out_channels[-1], self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(2*self.out_channels[-1],self.num_tasks)
	
        	


    def forward(self, batched_data):
        
      #  print(batched_data)
        h_node,h_select = self.gnn_node(batched_data)
        #print(h_node.shape)
        h_graph = self.pool(h_node, batched_data.batch)
        #print(h_graph.shape)
        
        # Compute the weighted sum of x_batch and x_sum
        w1 = 1.0  # Adjust this value as needed
        #h_out = w1 * h_graph + (1 - w1) * h_select
        #
        h_weight = w1 * (h_graph-h_select) 
        if h_select.dim() == 1:
            h_select = h_select.unsqueeze(0)
        h_out = torch.cat((h_select, h_weight), dim=1)

        #print(h_out.shape)
        #out = F.relu(self.lin1(out))
        #out=F.relu(self.graph_pred_linear(h_out))
        p=torch.nn.LeakyReLU(0.1)
        out=p(self.graph_pred_linear(h_out))
        #print(out.shape)
        
        return  out
    
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
    def __init__(self, num_layer, emb_dim,in_channels,out_channels,drop_ratio = 0.5, gnn_type = 'gin',heads=None):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.num_layer=num_layer
        self.heads=heads
        ### add residual connection or not


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")


        ###List of GNNs
#         self.convs = torch.nn.ModuleList()
#         self.batch_norms = torch.nn.ModuleList()

#         for i, in_c, out_c in zip(range(num_layer), 
#                                     in_channels, out_channels):
#             if gnn_type == 'gin':
#                 mlp = MLP([in_c, in_c, out_c])
#                 self.convs.append(GINConv(nn=mlp, train_eps=False))
                
#             elif gnn_type == 'gcn':
#                 #in_channels = hidden_channels
#                 self.convs.append(GCNConv(in_c,out_c))
             
#             elif gnn_type =='gat':
#                 if i==1:
#                     self.convs.append(GATv2Conv(int(in_c), out_c, heads=int(heads)))    
#                 else:
#                     self.convs.append(GATv2Conv(int(in_c*heads), out_c, heads=int(heads))) 
#             else:
#                 ValueError('Undefined GNN type called {}'.format(gnn_type))
                
#            # self.batch_norms.append(torch.nn.BatchNorm1d(embed))

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for i, in_c, out_c in zip(range(num_layer), in_channels, out_channels):
            if gnn_type == 'gin':
                mlp = MLP([in_c, in_c, out_c])
                self.convs.append(GINConv(nn=mlp, train_eps=False))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(in_c, out_c))
            elif gnn_type == 'gat':
                if i == 1:
                    self.convs.append(GATv2Conv(int(in_c), out_c, heads=int(heads)))
                else:
                    self.convs.append(GATv2Conv(int(in_c * heads), out_c, heads=int(heads)))
            # elif gnn_type='mpnn':
            #     nn = Sequential(Linear(in_c, in_c), ReLU(), Linear(in_c, out_c * out_c))
            #     self.convs.append (NNConv(in_c, in_c, nn))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(out_c))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x.float(), batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        batch_index=(batched_data.index).unsqueeze(1)
        edge_weight=None

        ### computing input node embedding
        #print('here')
        h_list = [x]
        #print(x.shape)
        for layer in range(self.num_layer):
            #print(layer)

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            #print(layer,h.shape)
            h = self.batch_norms[layer](h)
            #print(h.shape)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)
        
        

            # for layer in range(self.num_layer):
            #     node_representation += h_list[layer]
                
        #considering the last layer representation        
        node_representation = h_list[-1]
        
        node_select=node_representation[batch_index.squeeze()]

        return node_representation,node_select

    