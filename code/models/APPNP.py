from datasets import BasicDataset
from .LightGCN import LightGCN
from torch import nn
import torch.nn.functional as F
import scipy.sparse as sp
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


class PPRPowerIteration(nn.Module):
    def __init__(self, norm_adj_matrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        self.register_buffer('A_hat', (1 - alpha) * norm_adj_matrix.to_sparse())
        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, E: torch.FloatTensor):
        preds = E
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * E
        return preds


class APPNP(LightGCN):
    """Extension to use APPNP with LightGCN"""

    def __init__(self, config: dict, dataset: BasicDataset):
        super().__init__(config, dataset)
        n_nodes = self.adj_mat.shape[0]
        norm_adj = self.dataset.normalize_adj(self.adj_mat + sp.eye(n_nodes))
        self.propagation = PPRPowerIteration(norm_adj, alpha=config['alpha'], niter=config['num_walks'])

        # To save memory
        del norm_adj

    def forward(self):
        all_users_embeddings, all_items_embeddings = super().forward()

        light_out = torch.vstack((all_users_embeddings, all_items_embeddings))

        # Approximate personalized propagation of neural predictions
        light_out = self.propagation(light_out)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings
