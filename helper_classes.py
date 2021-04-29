import torch
import math
import torch.nn as nn
import numpy as np

class CosineCutoff(nn.Module):

    def __init__(self, cutoff=1.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

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
    
class BesselBasis(nn.Module):
    

    def __init__(self, cutoff=1.0, n_rbf=None):
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
    
class Prepare_Message_Vector(nn.Module):

    def __init__(self, num_nodes):
        super(Prepare_Message_Vector, self).__init__()
        self.num_nodes = num_nodes

    def forward(self,s,v):
    
        ''' Takes inputs s and v and prepares a vector for message passing by flattening and 
        concatenating. The vector is repeated for the total number of nodes. Works only if every 
        node is initialized the same way.
    
        Args:
            s: torch.tensor(N,1)
            v: torch.tensor(F,3)
        Returns:
            troch.tensor(num_nodes,(Fx3)+N,1), flat_shape_v, flat_shape_s
         '''
    
        flat_shape_v = v.flatten().shape
        flat_shape_s = s.flatten().shape
    
        message_vector = torch.cat((s.flatten(), v.flatten()))
        message_vector = message_vector.repeat(self.num_nodes,1)
    
    
        return message_vector, flat_shape_v[0], flat_shape_s[0]
    
