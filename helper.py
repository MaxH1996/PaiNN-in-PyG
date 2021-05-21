import torch 
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, radius_graph, radius
from torch_geometric.utils import add_self_loops, degree
from deepqmc.torchext import SSP, get_log_dnn
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader

import math
import torch.nn as nn
import torch.nn.functional as Func
from torch.nn import Embedding, Sequential, Linear, ModuleList, Module
import numpy as np


class CosineCutoff(torch.nn.Module):

    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        #self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.cutoff = cutoff

    def forward(self, distances):
        """Compute cutoff.

        Args:
            distances (torch.Tensor): values of interatomic distances.

        Returns:
            torch.Tensor: values of cutoff function.

        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs
    
class BesselBasis(torch.nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff=5.0, n_rbf=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        inputs = torch.norm(inputs, p=2, dim=1)
        a = self.freqs
        ax = torch.outer(inputs,a)
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[:,None]

        return y
    
class Jastrow(nn.Module):

    def __init__(
        self, embedding_dim, activation_factory=SSP, *, n_layers=3, sum_first=True
    ):
        super().__init__()
        self.net = get_log_dnn(embedding_dim, 1, activation_factory, n_layers=n_layers)
        self.sum_first = sum_first

    def forward(self, xs):
        if self.sum_first:
            xs = self.net(xs.sum(dim=-2))
        else:
            xs = self.net(xs).sum(dim=-2)
        return xs.squeeze(dim=-1)
    
class Bipartite(Data):
    def __init__(self, edge_index, coord_elec, coord_nuc,s_nuc,v_nuc,num_nodes):
        super(Bipartite, self).__init__()
        self.edge_index = edge_index
        self.coord_elec = coord_elec
        self.coord_nuc = coord_nuc
        self.s_nuc = s_nuc
        self.v_nuc = v_nuc
        self.num_nodes = num_nodes
    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.coord_nuc.size(0)], [self.coord_elec.size(0)]])
        else:
            return super().__inc__(key, value)
        
class BatchGraphNuc(nn.Module):
    def __init__(self, cut_off):
        super(BatchGraphNuc, self).__init__()
        self.dim = cut_off
        
    def forward(self,  s_nuc,v_nuc, coord_elec, coord_nuc):
        
        batch_dim, n_elec = coord_elec.shape[:2]
        
        coord_nuc = coord_nuc.repeat(batch_dim,1,1)
        
        data_list = [Bipartite(radius(e,n,5.0),e,n,sn,vn, n_elec) 
                     for e,n,sn,vn in zip(coord_elec, coord_nuc, s_nuc,v_nuc)]
        
        loader = DataLoader(data_list, batch_size=batch_dim)
        batch = next(iter(loader))
        
        row, col = batch.edge_index
        edge_attr = batch.coord_elec[col] - batch.coord_nuc[row]
        
        return (batch.s_nuc, batch.v_nuc, batch.edge_index, edge_attr)
    
class BatchGraphElec(nn.Module):

    def __init__(self,cut_off=5.0):
        super(BatchGraphElec, self).__init__()
        self.cut_off = cut_off

    def forward(self,s, v, rs):
        # rs are converted to edge_attributes
        # num_elec = num_nodes
        batch_dim, n_elec = rs.shape[:2] 
        data = Batch.from_data_list([Data(x=s, v=v, r=r) for s, v, r in zip(s, v, rs)])
        
        
        batch_edge_index = radius_graph(data.r, r=self.cut_off, batch=data.batch, loop=False)

        batch_row, batch_col = batch_edge_index
        batch_edge_attr = data.r[batch_row] - data.r[batch_col]

        return data.x, data.v, batch_edge_index, batch_edge_attr
    
class BackflowPaiNN(nn.Module):


    def __init__(
        self,
        embedding_dim,
        n_backflows,
        num_electrons
    ):
        super().__init__()
         
        self.net = nn.Sequential(
            Linear(embedding_dim, embedding_dim),
            Linear(embedding_dim, 1)
        )
            

    def forward(self, xs):
        return torch.squeeze(self.net(xs))
