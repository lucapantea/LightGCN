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
    def __init__(self, config: dict, dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.sigmoid = nn.Sigmoid()
        self.__init_weight()

    def get_embedding_matrix(self):
        return torch.vstack((
            self.embedding_user.weight.data,
            self.embedding_item.weight.data
        )).unsqueeze(1)

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def get_user_rating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())

        return self.sigmoid(scores)

    def forward(self, users, items):
        users = users.long()
        items = items.long()

        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)

        scores = torch.sum(users_emb*items_emb, dim=1)

        return self.sigmoid(scores)
