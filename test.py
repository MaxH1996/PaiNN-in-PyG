import torch
import torch.nn as nn
import torch.nn.functional as Func

from PaiNN import PaiNN

# Paramerts
# F: Num. features, r_ij: cartesian positions
F = 128
num_nodes = 4
s0 = torch.rand(num_nodes, F, dtype=torch.float)
v0 = torch.zeros(num_nodes, F, 3, dtype=torch.float)
r_ij = torch.tensor(
    [
        [0.000000, 0.000000, -0.537500],
        [0.000000, 0.000000, 0.662500],
        [0.000000, 0.866025, -1.037500],
        [0.000000, -0.866025, -1.037500],
    ]
)


# edge_attr: inter_atomic distances
edge_index = radius_graph(r_ij, r=1.30, batch=None, loop=False)
row, col = edge_index
edge_attr = r_ij[row] - r_ij[col]
# print(edge_index.dtype == torch.long)

if __name__ == "__main__":
    PA = PaiNN(F, F, 4)
    form = PA(s0, v0, edge_index, edge_attr)
    print(form[0])
