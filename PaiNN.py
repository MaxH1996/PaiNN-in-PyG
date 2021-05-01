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

from helper import CosineCutoff, BesselBasis, Prepare_Message_Vector
from message import MessagePassPaiNN
from update import UpdatePaiNN

class PaiNN(torch.nn.Module):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20, num_interactions=3):
        super(PaiNN, self).__init__() 
        '''PyG implementation of PaiNN network of Schütt et. al. Supports two arrays  
           stored at the nodes of shape (num_feat,1) and (num_feat,3). For this representation
           to be compatible with PyG, the arrays are flattened and concatenated. 
           Important to note is that the out_channels must match number of features'''
        
        self.num_nodes = num_nodes
        self.num_interactions = num_interactions
        self.cut_off = cut_off
        self.n_rbf = n_rbf
        self.prepare = Prepare_Message_Vector(num_nodes)
        
        self.Message = MessagePassPaiNN(num_feat, out_channels, num_nodes, cut_off, n_rbf)
        self.Update = UpdatePaiNN(num_feat, out_channels, num_nodes)


    def forward(self, s,v, edge_index, edge_attr):
        
        # First interaction loop. No addition to update from initial s_i
        x, flat_shape_v, flat_shape_s = self.prepare(s,v)
        x = self.Message(x, edge_index, edge_attr,flat_shape_s, flat_shape_v) 
        feed_in = torch.cat((torch.zeros(flat_shape_s), v.flatten()))
        x = self.Update(x+feed_in, flat_shape_s, flat_shape_v) + x
        
        for i in range(self.num_interactions-1):
            
            x = self.Message(x, edge_index, edge_attr,flat_shape_s, flat_shape_v) + x
            x = self.Update(x,flat_shape_s, flat_shape_v) + x 
         
        
        
        s,v = torch.split(x, [flat_shape_s, flat_shape_v], dim=-1)
        return s, v.reshape(-1, int(flat_shape_v/3), 3)