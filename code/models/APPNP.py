from datasets import BasicDataset
from .LightGCN import LightGCN
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import torch


class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)

class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)

def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices),
        torch.FloatTensor(coo.data),
        coo.shape)

def calc_A_hat(adj_matrix):
    adj_matrix = sp.csr_matrix(adj_matrix.to_dense().numpy())
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1 # degree matrix
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))
        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, E: torch.FloatTensor):
        print('')
        preds = E
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * E
        return preds


class APPNP(LightGCN):
    """Extension to use APPNP with LightGCN"""

    def __init__(self, config: dict, dataset: BasicDataset):
        super().__init__(config, dataset)
        self.propagation = PPRPowerIteration(self.graph, alpha=0.1, niter=10)

    def forward(self):
        all_users_embeddings, all_items_embeddings = super().forward()

        light_out = torch.vstack((all_users_embeddings, all_items_embeddings))

        # Approximate personalized propagation of neural predictions
        light_out = self.propagation(light_out)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings
