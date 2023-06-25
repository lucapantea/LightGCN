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
    def __init__(self):
        super(BasicModel, self).__init__()

    def get_user_rating(self, users):
        raise NotImplementedError

    def parameters_norm(self):
        return torch.tensor(0)
