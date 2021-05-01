import torch 
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import add_self_loops, degree

import ase
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn import Embedding, Sequential, Linear, ModuleList, Module
import numpy as np
import math

from helper import CosineCutoff, BesselBasis, Prepare_Message_Vector

class MessagePassPaiNN(MessagePassing):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20):
        super(MessagePassPaiNN, self).__init__(aggr='add') 
        
        self.lin1 = Linear(num_feat, out_channels) 
        self.lin2 = Linear(out_channels, 3*out_channels) 
        self.lin_rbf = Linear(n_rbf, 3*out_channels) 
        self.silu = Func.silu
        
        self.prepare = Prepare_Message_Vector(num_nodes)
        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)  
    
    def forward(self, x, edge_index, edge_attr, flat_shape_s,flat_shape_v):
        
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr
                            ,flat_shape_s=flat_shape_s, flat_shape_v=flat_shape_v)
            
        return x    
    
    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v):
        
        
        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)
        
        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        W = torch.einsum('ij,i->ij',ch1,edge_attr.norm(dim=-1)) # ch1 * f_cut
        
        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)
        
        # Split 
        left, dsm, right = torch.tensor_split(phi*W,3,dim=-1)
        
        # v_j channel
        normalized = Func.normalize(edge_attr, p=2, dim=1)
        v_j = v_j.reshape(-1, int(flat_shape_v/3), 3)
        hadamard_right = torch.einsum('ij,ik->ijk',right, normalized)
        hadamard_left = torch.einsum('ijk,ij->ijk',v_j,left)
        dvm = hadamard_left + hadamard_right 
        
        # Prepare vector for update
        x_j = torch.cat((dsm,dvm.flatten(-2)), dim=-1)
       
        return x_j