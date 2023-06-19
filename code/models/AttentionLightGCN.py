import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .LightGCN import LightGCN
from datasets import BasicDataset


class ScaledDotProductAttentionLightGCN(LightGCN):
    """Extending LightGCN by adding scaled dot-product attention."""

    def forward(self):
        """Forward pass of the model."""
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                print("dropping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph

        for _ in range(self.n_layers):
            if self.a_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))

                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        if self.config['save_embs']:
            self.embs = embs

        queries, keys, values = self.prepare_attention_inputs(embs)
        attention_output = self.compute_attention(queries, keys, values)

        all_users_embeddings, all_items_embeddings = torch.split(
            attention_output[:, -1, :], [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings

    def prepare_attention_inputs(self, embs):
        return embs, embs, embs

    @staticmethod
    def compute_attention(queries, keys, values):
        scaling_factor = math.sqrt(queries.size(-1))
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scaling_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output


class WeightedScaledDotProductAttentionLightGCN(ScaledDotProductAttentionLightGCN):
    """Extending ScaledDotProductAttentionLightGCN by adding linear projections to embeddings."""

    def __init__(self, config: dict, dataset: BasicDataset):
        super().__init__(config, dataset)
        self.latent_dim = self.config['latent_dim_rec']
        self.attention_dim = self.config.get('attention_dim', 1)
        self.query_projection = nn.Linear(self.latent_dim, self.attention_dim)
        self.key_projection = nn.Linear(self.latent_dim, self.attention_dim)
        self.value_projection = nn.Linear(self.latent_dim, self.attention_dim)

    def prepare_attention_inputs(self, embs):
        queries = self.query_projection(embs)
        keys = self.key_projection(embs)
        values = self.value_projection(embs)
        return queries, keys, values
