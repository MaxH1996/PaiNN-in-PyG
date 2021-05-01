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


class UpdatePaiNN(torch.nn.Module):
    def __init__(self, num_feat, out_channels, num_nodes):
        super(UpdatePaiNN, self).__init__() 
        
        self.lin_up = Linear(2*num_feat, out_channels) 
        self.denseU = Linear(3,out_channels, bias = False) 
        self.denseV = Linear(3,out_channels, bias = False) 
        self.lin2 = Linear(out_channels, 3*out_channels) 
        self.silu = Func.silu
        
        
    def forward(self, out_aggr, flat_shape_s, flat_shape_v):
        
        # split and take linear combinations
        s, v = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)
        
        v_u = v.reshape(-1, int(flat_shape_v/3), 3)
        U = self.denseU(v_u)
        V = self.denseV(v_u)
        
        # form the dot product
        UV =  torch.einsum('ijk,ijk->ij',U,V) 
        
        # s_j channel
        nV = torch.norm(V, dim=-1)

        s_u = torch.cat([s, nV], dim=-1)
        s_u = self.lin_up(s_u) 
        s_u = Func.silu(s_u)
        s_u = self.lin2(s_u)
        s_u = Func.silu(s_u)
        
        # final split
        top, middle, bottom = torch.tensor_split(s_u,3,dim=-1)
        
        # outputs
        dvu = torch.einsum('ijk,ij->ijk',v_u,Func.silu(top)) 
        dsu = Func.silu(middle)*UV + Func.silu(bottom) 
        
        update = torch.cat((dsu,dvu.flatten(-2)), dim=-1)
        
        return update