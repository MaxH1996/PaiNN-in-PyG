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

class MessagePassPaiNN(MessagePassing):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20):
        super(MessagePassPaiNN, self).__init__(aggr='add') 
        
        self.lin1 = Linear(num_feat, out_channels) 
        self.lin2 = Linear(out_channels, 3*out_channels) 
        self.lin_rbf = Linear(n_rbf, 3*out_channels) 
        self.silu = Func.silu
        
        
        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)
        self.num_nodes = num_nodes
        self.num_feat = num_feat
    
    def forward(self, s,v, edge_index, edge_attr):
        
        s = s.flatten(-1)
        v = v.flatten(-2)
        
        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]
    
        x =torch.cat([s, v], dim = -1)
        
        
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr
                            ,flat_shape_s=flat_shape_s, flat_shape_v=flat_shape_v)
            
        return x    
    
    def message(self, x_j, edge_attr, flat_shape_s, flat_shape_v):
        
        
        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)
        
        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        cut = self.f_cut(edge_attr.norm(dim=-1))
        W = torch.einsum('ij,i->ij',ch1, cut) # ch1 * f_cut
        
        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)
        
        # Split 
        
        left, dsm, right = torch.split(phi*W, self.num_feat, dim=-1)
        
        
        # v_j channel
        normalized = Func.normalize(edge_attr, p=2, dim=1)
        v_j = v_j.reshape(-1, int(flat_shape_v/3), 3)
        hadamard_right = torch.einsum('ij,ik->ijk',right, normalized)
        hadamard_left = torch.einsum('ijk,ij->ijk',v_j,left)
        dvm = hadamard_left + hadamard_right 
        
        # Prepare vector for update
        x_j = torch.cat((dsm,dvm.flatten(-2)), dim=-1)
       
        return x_j
    
    def update(self, out_aggr,flat_shape_s, flat_shape_v):
        
        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)
        
        return s_j, v_j.reshape(-1, int(flat_shape_v/3), 3)
    
    
    class MessagePassPaiNN_NE(MessagePassing):
    def __init__(self, num_feat, out_channels, num_nodes, cut_off=5.0, n_rbf=20):
        super(MessagePassPaiNN_NE, self).__init__(aggr='add') 
        
        self.lin1 = Linear(num_feat, out_channels) 
        self.lin2 = Linear(out_channels, 3*out_channels) 
        self.lin_rbf = Linear(n_rbf, 3*out_channels) 
        self.silu = Func.silu
        
        #self.prepare = Prepare_Message_Vector(num_nodes)
        self.RBF = BesselBasis(cut_off, n_rbf)
        self.f_cut = CosineCutoff(cut_off)
        self.num_nodes = num_nodes
        self.num_feat = num_feat
    
    def forward(self, s,v,s_nuc,v_nuc, edge_index, edge_attr):
        
        s = s.flatten(-1)
        v = v.flatten(-2)
        
        s_nuc = s_nuc.flatten(-1)
        v_nuc = v_nuc.flatten(-2)
        
        flat_shape_v = v.shape[-1]
        flat_shape_s = s.shape[-1]
        
        n_nuc = s_nuc.shape[0]
        n_elec = s.shape[0]
        
        x_p =torch.cat([s_nuc, v_nuc], dim = -1) # nuclei
        x = torch.cat([s, v], dim = -1) # electrons
        
        x = self.propagate(edge_index,x=(x_p,x), edge_attr=edge_attr
                            ,flat_shape_s=flat_shape_s, flat_shape_v=flat_shape_v, size=(n_nuc,n_elec))
            
        return x    
    
    def message(self,x_j, edge_attr, flat_shape_s, flat_shape_v):
        
        # Split Input into s_j and v_j
        s_j, v_j = torch.split(x_j, [flat_shape_s, flat_shape_v], dim=-1)
        #_, v_i = torch.split(x_i, [flat_shape_s, flat_shape_v], dim=-1)
        
        
        # r_ij channel
        rbf = self.RBF(edge_attr)
        ch1 = self.lin_rbf(rbf)
        cut = self.f_cut(edge_attr.norm(dim=-1))
        W = torch.einsum('ij,i->ij',ch1, cut) # ch1 * f_cut
        
        # s_j channel
        phi = self.lin1(s_j)
        phi = self.silu(phi)
        phi = self.lin2(phi)
        
        # Split 
        left, dsm, right = torch.split(phi*W, self.num_feat, dim=-1)
        
        # v_j channel
        normalized = Func.normalize(edge_attr, p=2, dim=1)
        
        v_j = v_j.reshape(-1, int(flat_shape_v/3), 3)
        #v_i = v_i.reshape(-1, int(flat_shape_v/3), 3)
        #print(v_j - v_i)
        hadamard_right = torch.einsum('ij,ik->ijk',right, normalized)
        hadamard_left = torch.einsum('ijk,ij->ijk',v_j,left)
        dvm = hadamard_left + hadamard_right 
        
        # Prepare vector for update
        x_j = torch.cat((dsm,dvm.flatten(-2)), dim=-1)
       
        return x_j
    
    def update(self, out_aggr,flat_shape_s, flat_shape_v):
        
        s_j, v_j = torch.split(out_aggr, [flat_shape_s, flat_shape_v], dim=-1)
        
        return s_j, v_j.reshape(-1, int(flat_shape_v/3), 3)
