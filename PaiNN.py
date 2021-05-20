import torch 
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import add_self_loops, degree

import ase
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn import Embedding, Sequential, Linear, ModuleList, Module
import numpy as np
from torch import linalg as LA
import math

from helper import CosineCutoff, BesselBasis
from message import MessagePassPaiNN, MessagePassPaiNN_NE
from update import UpdatePaiNN

class PaiNN(torch.nn.Module):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20, num_interactions=3):
        super(PaiNN, self).__init__() 
        '''PyG implementation of PaiNN network of Schütt et. al. Supports two arrays  
           stored at the nodes of shape (num_nodes,num_feat,1) and (num_nodes, num_feat,3). For this 
           representation to be compatible with PyG, the arrays are flattened and concatenated. 
           Important to note is that the out_channels must match number of features'''
        
        
        self.num_interactions = num_interactions
        self.cut_off = cut_off
        self.n_rbf = n_rbf
        self.num_nodes = num_nodes
        self.num_feat = num_feat
        self.out_channels = out_channels
        self.lin = Linear(num_feat, num_feat)
        self.silu = Func.silu
        
        
        self.list_message = nn.ModuleList(
            [
                MessagePassPaiNN(num_feat, out_channels, num_nodes, cut_off, n_rbf)
                for _ in range(self.num_interactions)
            ]
        )
        self.list_update = nn.ModuleList(
            [
                UpdatePaiNN(num_feat, out_channels, num_nodes)
                for _ in range(self.num_interactions)
            ]
        )
        
        


    def forward(self, s,v, edge_index, edge_attr):
        
        
        for i in range(self.num_interactions):
            
            s_temp,v_temp = self.list_message[i](s,v, edge_index, edge_attr)
            s, v = s_temp+s, v_temp+v
            s_temp,v_temp = self.list_update[i](s,v) 
            s, v = s_temp+s, v_temp+v 
        
        s=self.lin(s)
        s=self.silu(s)
        s=self.lin(s)
        
        
        return s
    
class PaiNNElecNuc(torch.nn.Module):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20, num_interactions=3):
        super(PaiNNElecNuc, self).__init__() 
        '''PyG implementation of PaiNN network of Schütt et. al. Supports two arrays  
           stored at the nodes of shape (num_nodes,num_feat,1) and (num_nodes, num_feat,3). For this 
           representation to be compatible with PyG, the arrays are flattened and concatenated. 
           Important to note is that the out_channels must match number of features'''
        
        self.num_nodes = num_nodes
        self.num_interactions = num_interactions
        self.cut_off = cut_off
        self.n_rbf = n_rbf
        self.linear = Linear(num_feat,num_feat)
        self.silu = Func.silu
        
        self.list_message = nn.ModuleList(
            [
                MessagePassPaiNN(num_feat, out_channels, num_nodes, cut_off, n_rbf)
                for _ in range(self.num_interactions)
            ]
        )
        self.list_update = nn.ModuleList(
            [
                UpdatePaiNN(num_feat, out_channels, num_nodes)
                for _ in range(self.num_interactions)
            ]
        )

        self.list_message_NE = nn.ModuleList(
            [
                MessagePassPaiNN_NE(num_feat, out_channels, num_nodes)
                for _ in range(self.num_interactions)
            ]
        )
    def forward(self, s,v, s_nuc,v_nuc, edge_index, edge_attr, edge_index_nuc, edge_attr_nuc):
        
        
        for i in range(self.num_interactions):
            
            s_temp,v_temp = self.list_message[i](s,v, edge_index, edge_attr)
            s_temp_NE, v_temp_NE = self.list_message_NE[i](s,v, s_nuc,v_nuc, 
                                                           edge_index_nuc, edge_attr_nuc)
            
            s, v = s_temp + s + s_temp_NE, v_temp + v + v_temp_NE
            s_temp,v_temp = self.list_update[i](s,v) 
            s, v = s_temp+s, v_temp+v
            
        
        s = self.linear(s)
        s = self.silu(s)
        s = self.linear(s)
        
        return s, v