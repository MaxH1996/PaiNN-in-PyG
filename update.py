import torch 
from torch_geometric.nn import MessagePassing, radius_graph
from torch_geometric.utils import add_self_loops, degree

import ase
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn import Embedding, Sequential, Linear, ModuleList, Module
import numpy as np
import math

from helper import CosineCutoff, BesselBasis


class UpdatePaiNN(torch.nn.Module):
    def __init__(self, num_feat, out_channels, num_nodes):
        super(UpdatePaiNN, self).__init__() 
        
        self.lin_up = Linear(2*num_feat, out_channels) 
        self.denseU = Linear(num_feat,out_channels, bias = False) 
        self.denseV = Linear(num_feat,out_channels, bias = False) 
        self.lin2 = Linear(out_channels, 3*out_channels) 
        self.silu = Func.silu
        self.num_feat = num_feat
        
        
    def forward(self, s,v):
        
        
        s = s.flatten(-1)
        v = v.flatten(-2)
        
        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]
        
        v_u = v.reshape(-1, int(flat_shape_v/3), 3)
        v_ut = torch.transpose(v_u,1,2) # need transpose to get lin.comb a long feature dimension
        U = torch.transpose(self.denseU(v_ut),1,2)
        V = torch.transpose(self.denseV(v_ut),1,2)
        
        
        # form the dot product
        UV =  torch.einsum('ijk,ijk->ij',U,V) 
        
        # s_j channel
        nV = torch.norm(V, dim=-1)

        s_u = torch.cat([s, nV], dim=-1)
        s_u = self.lin_up(s_u) 
        s_u = Func.silu(s_u)
        s_u = self.lin2(s_u)
        #s_u = Func.silu(s_u)
        
        # final split
        top, middle, bottom = torch.split(s_u, self.num_feat, dim=-1)
        
        
        # outputs
        dvu = torch.einsum('ijk,ij->ijk',v_u,top) 
        dsu = middle*UV + bottom 
        
        
        return dsu, dvu.reshape(-1, int(flat_shape_v/3), 3)
