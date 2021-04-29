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
from main_painn import PaiNN


#Paramerts
# F: Num. features, r_ij: cartesian positions
F = 128
num_nodes = 4
s0 = torch.rand(F, dtype=torch.float)
v0 = torch.zeros(F,3, dtype=torch.float)
r_ij =  torch.tensor([[0.000000,  0.000000,  -0.537500],
  [0.000000,  0.000000,   0.662500],
  [0.000000,  0.866025,  -1.037500],
  [0.000000, -0.866025,  -1.037500]]) 

# edge_attr: inter_atomic distances
edge_index = radius_graph(r_ij, r=1.30, batch=None, loop=False)
row, col = edge_index
edge_attr = (r_ij[row] - r_ij[col])
#print(edge_index.dtype == torch.long)

PA = PaiNN(F, F, 4)
s_final, v_final = PA(s0,v0, edge_index,edge_attr)
print(v_final)
print(v_final)


