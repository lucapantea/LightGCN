"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
from .BasicModel import BasicModel
from datasets import BasicDataset
from torch import nn
import torch


class PureMF(BasicModel):
    """
    Pure Matrix Factorization (PureMF) model for recommendation.

    Args:
        config (dict): Configuration dictionary.
        dataset (BasicDataset): The dataset object.

    Attributes:
        num_users (int): Number of users in the dataset.
        num_items (int): Number of items in the dataset.
        latent_dim (int): Dimensionality of the latent embeddings.
        sigmoid (torch.nn.Sigmoid): Sigmoid activation function.
        embedding_user (torch.nn.Embedding): Embedding layer for users.
        embedding_item (torch.nn.Embedding): Embedding layer for items.

    Methods:
        get_embedding_matrix(): Get the embedding matrix.
        __init_weight(): Initialize the model's weights.
        get_user_rating(users): Get predicted ratings for the given users.
        forward(users, items): Forward pass of the model.

    Inherits from:
        BasicModel
    """

    def __init__(self, config: dict, dataset: BasicDataset):
        """
        Initialize the PureMF model.

        Args:
            config (dict): Configuration dictionary.
            dataset (BasicDataset): The dataset object.
        """
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.sigmoid = nn.Sigmoid()
        self.__init_weight()

    def get_embedding_matrix(self):
        """Get the embedding matrix."""
        return torch.vstack((
            self.embedding_user.weight.data,
            self.embedding_item.weight.data
        )).unsqueeze(1)

    def __init_weight(self):
        """Initialize the model's weights."""
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def get_user_rating(self, users):
        """Get predicted ratings for the given users."""
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())

        return self.sigmoid(scores)

    def forward(self, users, items):
        """Forward pass of the model."""
        users = users.long()
        items = items.long()

        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)

        scores = torch.sum(users_emb*items_emb, dim=1)

        return self.sigmoid(scores)
