"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import torch


class BPRLoss(object):
    """
    BPR loss function for pairwise ranking.

    Args:
        weight_decay (float): Weight decay coefficient.

    Methods:
        __call__(users_embeddings, pos_items_embeddings, neg_items_embeddings,
                 users_embeddings_layer0, pos_items_embeddings_layer0, 
                 neg_items_embeddings_layer0):
            Compute the BPR loss given the embeddings.

    """

    def __init__(self, weight_decay):
        """
        Initialize the BPRLoss.

        Args:
            weight_decay (float): Weight decay coefficient.
        """
        self.weight_decay = weight_decay

    def __call__(self,
                 users_embeddings,
                 pos_items_embeddings,
                 neg_items_embeddings,
                 users_embeddings_layer0,
                 pos_items_embeddings_layer0,
                 neg_items_embeddings_layer0,
                 parameters_norm):
        """
        Compute the BPR loss given the embeddings.

        Args:
            users_embeddings: Embeddings of the users.
            pos_items_embeddings: Embeddings of the positive items.
            neg_items_embeddings: Embeddings of the negative items.
            users_embeddings_layer0: Embeddings of the users in the first
                                     layer.
            pos_items_embeddings_layer0: Embeddings of the positive items in
                                         the first layer.
            neg_items_embeddings_layer0: Embeddings of the negative items in
                                         the first layer.

        Returns:
            torch.Tensor: Computed BPR loss.
        """
        reg_loss = (1 / 2) * (users_embeddings_layer0.norm(2).pow(2) +
                              pos_items_embeddings_layer0.norm(2).pow(2) +
                              neg_items_embeddings_layer0.norm(2).pow(2) +
                              parameters_norm
                              ) / users_embeddings.shape[0]

        pos_scores = torch.mul(users_embeddings, pos_items_embeddings)
        pos_scores = torch.sum(pos_scores, dim=1)

        neg_scores = torch.mul(users_embeddings, neg_items_embeddings)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(
            neg_scores - pos_scores))
        reg_loss *= self.weight_decay
        loss += reg_loss

        return loss
