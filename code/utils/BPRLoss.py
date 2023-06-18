"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
from models import BasicModel
from torch import optim
import torch


class BPRLoss(object):
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay

    def __call__(self,
                 users_embeddings,
                 pos_items_embeddings,
                 neg_items_embeddings,
                 users_embeddings_layer0,
                 pos_items_embeddings_layer0,
                 neg_items_embeddings_layer0):

        reg_loss = (1 / 2) * (users_embeddings_layer0.norm(2).pow(2) +
                              pos_items_embeddings_layer0.norm(2).pow(2) +
                              neg_items_embeddings_layer0.norm(2).pow(2)
                              ) / users_embeddings.shape[0]

        pos_scores = torch.mul(users_embeddings, pos_items_embeddings)
        pos_scores = torch.sum(pos_scores, dim=1)

        neg_scores = torch.mul(users_embeddings, neg_items_embeddings)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(
            torch.nn.functional.softplus(neg_scores - pos_scores))
        reg_loss *= self.weight_decay

        loss += reg_loss

        return loss
