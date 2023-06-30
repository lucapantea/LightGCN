"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
from torch import nn
import torch


class BasicModel(nn.Module):
    """
    BasicModel is a base model for developing custom recommendation systems.

    Methods:
        get_user_rating(users): It should return the ratings for given users.
        parameters_norm(): Returns the norm of the parameters in the model.
    """
    def __init__(self):
        """
        Initializes the BasicModel.
        """
        super(BasicModel, self).__init__()

    def get_user_rating(self, users):
        """
        Should return the ratings for the given users.

        Args:
            users: The users for whom to retrieve ratings.
        """
        raise NotImplementedError

    def parameters_norm(self):
        """
        Returns the norm of the parameters in the model.
        """
        return torch.tensor(0)
