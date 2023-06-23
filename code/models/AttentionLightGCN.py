from datasets import BasicDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .LightGCN import LightGCN
from datasets import BasicDataset


class BaseAttention(LightGCN):
    """Extending LightGCN by adding a simple attention mechanism."""

    def __init__(self, config: dict, dataset: BasicDataset):
        super().__init__(config, dataset)
        self.attention_weights = torch.nn.Parameter(torch.randn(config['lightGCN_n_layers'] + 1),
                                                    requires_grad=True)
        
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

        # Attention Mechanism
        attention_scores = F.softmax(self.attention_weights, dim=0)
        light_out = torch.sum(embs * attention_scores.view(1, -1, 1), dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings
    
class FinerAttention(LightGCN):
    """Extending LightGCN by adding a finer attention mechanism."""
    def __init__(self, config: dict, dataset: BasicDataset):
        super().__init__(config, dataset)
        self.attention_weights_users = torch.nn.Parameter(torch.randn(config['lightGCN_n_layers'] + 1),
                                                          requires_grad=True)
        self.attention_weights_items = torch.nn.Parameter(torch.randn(config['lightGCN_n_layers'] + 1),
                                                          requires_grad=True)
        
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

        # Calculate the softmax
        attention_scores_users = F.softmax(self.attention_weights_users, dim=0).view(1, -1, 1)
        attention_scores_items = F.softmax(self.attention_weights_items, dim=0).view(1, -1, 1)

        # Create block diagonal attention matrix
        attention_scores = torch.cat([attention_scores_users.repeat(self.num_users, 1, 1),
                                      attention_scores_items.repeat(self.num_items, 1, 1)], dim=0)

        # Compute weighted sum in one step
        light_out = torch.sum(embs * attention_scores, dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings
        

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

        if self.config['single']:
            light_out = attention_output[:, -1, :].squeeze()
        else:
            light_out = torch.mean(attention_output, dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

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
