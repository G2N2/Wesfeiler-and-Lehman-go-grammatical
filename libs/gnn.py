import torch
import torch.nn as nn
from libs.layer_gnn import G2N2Layer
from torch_geometric.nn import (global_add_pool,global_mean_pool, global_max_pool)
import libs.readout_gnn as ro

from time import time

def get_n_params(model):
    pp=0
    for p in model.parameters():
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class G2N2(nn.Module):
    def __init__(self,  node_input_dim, edge_input_dim, output_dim, device,
                 num_layer = 5, nodes_dim = [16,16,16,16,16], 
                 edges_dim = [16,16,16,16,16],decision_depth = 3,final_neuron = [512,256],
                 readout_type  = "sum" ,level = "graph"):
        
        super(G2N2, self).__init__()
        
        self.num_layer = num_layer
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        self.nodes_dim = nodes_dim
        self.edges_dim = edges_dim
        self.decision_depth = decision_depth
        self.final_neuron = final_neuron
        self.readout_type = readout_type
        self.level =level
        self.conv = nn.ModuleList()
        self.device = device
        
        if self.num_layer < 1:
            raise ValueError("Number of GNN layer must be greater than 1")
            
        if self.num_layer != len(self.nodes_dim):
            raise ValueError("Number of GNN layer must match length of nodes_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,len(self.nodes_dim)))
            
        if self.num_layer != len(self.edges_dim):
            raise ValueError("Number of GNN layer must match length of neuron_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,len(self.edges_dim)))
        if self.decision_depth != len(self.final_neuron) + 1:
            raise ValueError("Number of decision layer must match in decision depth" + 
                             "={}, final neuron dim + 1 = {}".format(self.decision_depth,len(self.final_neuron) + 1))
        
        if self.level == "edge":
            self.readout = ro.edge_level_readout
        elif self.level == "node":
            if self.readout_type == "sum":
                self.readout = ro.node_level_readout_sum
            elif self.readout_type == "mean":
                self.readout = ro.node_level_readout_mean
            elif self.readout_type == "max":
                self.readout = ro.node_level_readout_max
            else:
                raise ValueError("Invalid readout type")
        elif self.level == "graph":
            if self.readout_type == "sum":
                self.readout = ro.graph_level_readout_sum
            elif self.readout_type == "mean":
                self.readout = ro.graph_level_readout_mean
            elif self.readout_type == "max":
                self.readout = ro.graph_level_readout_max
            else:
                raise ValueError("Invalid readout type")
        else:
            raise ValueError("Invalid level type, should be graph,node or edge")
        
        for i in range(self.num_layer):
            if i == 0:
                
                self.conv.append(G2N2Layer(nedgeinput= edge_input_dim, nedgeoutput = self.edges_dim[0],
                                            nnodeinput= self.node_input_dim, nnodeoutput= self.nodes_dim[0], device = self.device))

            else:
                self.conv.append(G2N2Layer(nedgeinput= self.edges_dim[i-1], nedgeoutput = self.edges_dim[i],
                                            nnodeinput= self.nodes_dim[i-1], nnodeoutput= self.nodes_dim[i], device = self.device))
        if self.level == "graph" or level == "node":
            self.fc = nn.ModuleList( [torch.nn.Linear(self.nodes_dim[-1]+2*self.edges_dim[-1],self.final_neuron[0])])
        elif self.level == "edge":
            self.fc = nn.ModuleList( [torch.nn.Linear(2*self.edges_dim[-1],self.final_neuron[0])])
            
        else:
            raise ValueError("Invalid level type, should be graph,node or edge")
        
        for i in range(self.decision_depth-2):
            self.fc.append(torch.nn.Linear(self.final_neuron[i], self.final_neuron[i+1]))


        self.fc.append(torch.nn.Linear(self.final_neuron[-1], self.output_dim))
        
        
    
    def forward(self,data):
        x = data.x
        edge_index=data.edge_index2
        C=data.edge_attr
        identite = C[:,0:1]
        batch_node = data.batch
        
        for i,l in enumerate(self.conv):
            x,C=(l(x, edge_index, C,batch_node))
             if i < self.num_layer - 1:
                 x = torch.relu(x)
                 C = torch.relu(C)
        x = self.readout(x,C,data.batch,data.batch_edge,data.node_batch_edge,identite)
        for i in range(self.decision_depth-1):
            x = torch.relu(self.fc[i](x))
        return self.fc[-1](x) 
        
