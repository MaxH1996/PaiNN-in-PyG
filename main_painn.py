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

from torch_geometric.data import Data
from helper_classes import CosineCutoff, BesselBasis, Prepare_Message_Vector

class PaiNN(MessagePassing):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20, num_interactions=3):
        super(PaiNN, self).__init__(aggr='add') 
        
        self.lin1 = Linear(num_feat, out_channels) #128
        self.lin2 = Linear(out_channels, 3*out_channels) #384
        self.lin_rbf = Linear(n_rbf, 3*out_channels) #20x384
        self.lin_up = Linear(2*num_feat, out_channels) #256,128
        self.denseU = Linear(3,out_channels, bias = False) # Note: Not sure if this is equivalent to Kristof's code where 
        self.denseV = Linear(3,out_channels, bias = False) # lin.comb runs over the features. So input should be num_feat of denseU or denseV.
        self.silu = Func.silu
        self.num_interactions = num_interactions
        
        
        
        self.prepare = Prepare_Message_Vector(num_nodes)
        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)

    def forward(self, s,v, edge_index, edge_attr):
        
        # Prepare the data for messaging
        x, flat_shape_v, flat_shape_s = self.prepare(s,v)
        
        for i in range(self.num_interactions):
            # first interaction block does not take the previous s_j into account, but the subsquent do. 
            if i == 0:
                activate_s0 = False
            else:
                activate_s0 = True
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr
                              , flat_shape_s=flat_shape_s, flat_shape_v=flat_shape_v, 
                                       activate_s0 = activate_s0)
            
            
        s,v = torch.split(x, [flat_shape_s, flat_shape_v], dim=-1)
        return s, v.reshape(-1, int(flat_shape_v/3), 3)

    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v, activate_s0):
        
        
        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)
        
        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        W = self.f_cut(ch1)
        
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
        dvm = hadamard_left + hadamard_right + v_j # Also adding original v_j as in fig. 2(a)
        
        # adding initial s_j for all but the first loop
        if activate_s0 == False:
            dsm = dsm 
        else:
            dsm = dsm + s_j
        
        # Prepare vector for update
        x_j = torch.cat((dsm,dvm.flatten(-2)), dim=-1)
       
        return  x_j
    
    def update(self, out_aggr, flat_shape_s, flat_shape_v):
        
        # split and take linear combinations
        s, v = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)
        
        v_u = v.reshape(-1, int(flat_shape_v/3), 3)
        U = self.denseU(v_u)
        V = self.denseV(v_u)
        
        # form the dot product
        UV =  torch.einsum('ijk,ijk->ij',U,V) #U_dot@ torch.transpose(U_dot,1,2)
        
        # s_j channel
        nV = torch.norm(V, dim=-1)

        s_u = torch.cat([s, nV], dim=-1)
        s_u = self.lin_up(s_u) # replace by sequential
        s_u = Func.silu(s_u)
        s_u = self.lin2(s_u)
        s_u = Func.silu(s_u)
        
        # final split
        top, middle, bottom = torch.tensor_split(s_u,3,dim=-1)
        
        # outputs
        dvu = torch.einsum('ijk,ij->ijk',v_u,Func.silu(top)) + v_u # Also adding original v_u as in fig. 2(a)
        dsu = Func.silu(middle)*UV + Func.silu(bottom) + s # Also adding original s as in fig. 2(a)
        
        update = torch.cat((dsu,dvu.flatten(-2)), dim=-1)
        
        return update
    

