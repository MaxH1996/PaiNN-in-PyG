import torch
import torch.nn as nn

from helper import BackflowPaiNN, BatchGraphElec, BatchGraphNuc, Jastrow
from PaiNN import PaiNNElecNuc


class OmniPaiNN(nn.Module):
    def __init__(
        self,
        n_nuc,
        n_up,
        n_down,
        n_orbitals,
        n_backflows,
        *,
        embedding_dim=128,
        num_nodes=1,
        cut_off=5.0,
        n_rbf=20,
        num_interactions=3,
    ):

        super().__init__()
        self.ElectronicPaiNN = PaiNNElecNuc(embedding_dim, embedding_dim, num_nodes)

        self.jastrow = Jastrow(embedding_dim, sum_first=True)

        self.backflow = BackflowPaiNN(embedding_dim, n_backflows, n_up + n_down)

        self.batch_E = BatchGraphElec(cut_off)
        self.batch_N = BatchGraphNuc(cut_off)
        self.spin_idxs = torch.tensor(
            (n_up + n_down) * [0] if n_up == n_down else n_up * [0] + n_down * [1]
        )

        self.nuc_idxs = torch.arange(n_nuc)
        self.X = nn.Embedding(1 if n_up == n_down else 2, embedding_dim)
        self.Y = nn.Embedding(n_nuc, embedding_dim)

        self.eb = embedding_dim

    def forward(self, rs, rn):
        # Took out elect_dists and nuc_dists for try
        batch_dim, n_elec = rs.shape[:2]
        n_nuc = rn.shape[0]

        # Initializing Scalars and Vectors
        s_e = self.X(self.spin_idxs.repeat(batch_dim, 1))
        s_n = self.Y(self.nuc_idxs.repeat(batch_dim, 1))

        v_e = torch.zeros(batch_dim, n_elec, self.eb, 3, dtype=torch.float)
        v_n = torch.zeros(batch_dim, n_nuc, self.eb, 3, dtype=torch.float)

        # Creating Batches for e-e-Graph and e-N-Graph
        s_e, v_e, edge_index, edge_attr = self.batch_E(s_e, v_e, rs)
        s_n, v_n, edge_index_n, edge_attr_n = self.batch_N(s_n, v_n, rs, rn)

        scalars, vectors = self.ElectronicPaiNN(
            s_e, v_e, s_n, v_n, edge_index, edge_attr, edge_index_n, edge_attr_n
        )

        scalars = scalars.reshape(batch_dim, n_elec, -1)
        vectors = vectors.reshape(batch_dim, n_elec, self.eb, -1)

        jastrow = self.jastrow(scalars)
        backflow = self.backflow(torch.transpose(vectors, -2, -1))

        return jastrow, None, backflow
