"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
from .BasicModel import BasicModel
from datasets import BasicDataset
from torch import nn

import torch


class LightGCN(BasicModel):
    """
        LightGCN model for recommendation.

        Args:
            config (dict): Configuration dictionary.
            dataset (BasicDataset): The dataset object.

        Attributes:
            num_users (int): Number of users in the dataset.
            num_items (int): Number of items in the dataset.
            latent_dim (int): Dimensionality of the latent embeddings.
            n_layers (int): Number of layers in the LightGCN model.
            keep_prob (float): Keep probability for dropout.
            embedding_user (torch.nn.Embedding): Embedding layer for users.
            embedding_item (torch.nn.Embedding): Embedding layer for items.
            sigmoid (torch.nn.Sigmoid): Sigmoid activation function.
            graph (torch.sparse.FloatTensor): Sparse adjacency matrix.
            embs (torch.Tensor or None): Pre-trained embeddings if available.

        Methods:
            get_embedding_matrix(): Get the embedding matrix.
            __init_weight(): Initialize the model's weights.
            __dropout_x(x, keep_prob): Apply dropout to a sparse tensor.
            __dropout(keep_prob): Apply dropout to the adjacency matrix.
            forward(): Forward pass of the model.
            get_user_rating(users): Get predicted ratings for the given users.

        Inherits from:
            BasicModel
    """

    def __init__(self, config: dict, dataset: BasicDataset):
        """
        Initialize the LightGCN model.

        Args:
            config (dict): Configuration dictionary.
            dataset (BasicDataset): The dataset object.
        """
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.embs = None
        self.__init_weight()

    def get_embedding_matrix(self):
        """Get the embedding matrix."""
        return self.embs

    def __init_weight(self):
        """Initialize the model's weights, a.k.a the embedding matrix."""
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.a_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb']))
            print('use pretrained data')

        self.sigmoid = nn.Sigmoid()
        self.graph = self.dataset.get_sparse_graph()
        self.embs = None
        self.personalised_vecs = self.dataset.compute_personalized_vectors(self.graph, self.config['latent_dim_rec'], num_walks=1, walk_length=3)
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    @staticmethod
    def __dropout_x(x, keep_prob):
        """Apply dropout to a sparse tensor."""
        size = x.size()
        index = x.indices().t()
        values = x.values()

        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()

        index = index[random_index]
        values = values[random_index] / keep_prob

        g = torch.sparse.FloatTensor(index.t(), values, size)

        return g

    def __dropout(self, keep_prob):
        """Apply dropout to the adjacency matrix."""
        if self.a_split:
            graph = []
            for g in self.graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.graph, keep_prob)

        return graph

    def forward(self):
        """Forward pass of the model."""
        users_emb = self.embedding_user.weight
        personalized_emb = self.personalised_vecs
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        scaling_factor = 1 # Experiment w/ this value

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

            print('t')
            # Incorporate personalized vectors here
            personalized_emb_scaled = personalized_emb * scaling_factor  # Apply scaling factor

            with torch.no_grad():
                all_emb[:self.num_users, :] += personalized_emb_scaled

            print('t')

            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        if self.config['save_embs']:
            self.embs = embs

        if self.config['single']:
            light_out = embs[:, -1, :].squeeze()
        else:
            light_out = torch.mean(embs, dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings

    def get_user_rating(self, users):
        """Get predicted ratings for the given users."""
        all_users, all_items = self.forward()

        users_emb = all_users[users.long()]
        items_emb = all_items

        rating = self.sigmoid(torch.matmul(users_emb, items_emb.t()))

        return rating
