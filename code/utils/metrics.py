"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def novelty(ground_truth, train_items_interacted_batch, topk):
    novelty = 0.
    # We want to calculate the proportion of novel items recommended to a user
    for user_train_beh, user_recom in zip(ground_truth, train_items_interacted_batch):
        set_user_train_beh = set(user_train_beh)
        set_user_recom = set(user_recom)
        num_repeated_elements = len(set_user_train_beh.intersection(set_user_recom))
        novelty += (topk - num_repeated_elements) / topk
    return novelty


def mean_intra_list_distance(recommendation_lists, item_embeddings):
    # Get the embeddings of the recommended items
    recommended_embeddings = item_embeddings[recommendation_lists]

    # Compute pairwise distances
    dists = torch.cdist(recommended_embeddings, recommended_embeddings, p=2)

    # Since the distance matrix is symmetric, we take the upper triangular part excluding the diagonal
    upper_triangular_part = dists.triu(diagonal=1)

    # Compute the ILD
    if recommendation_lists.shape == 3:  # batched computation
        list_length = recommendation_lists.shape[2]
        ILDs = upper_triangular_part.sum(dim=[2, 3]) / (list_length * (list_length - 1) / 2)
    else:  # non-batched computation
        list_length = recommendation_lists.shape[1]
        ILDs = upper_triangular_part.sum(dim=[1, 2]) / (list_length * (list_length - 1) / 2)

    return torch.sum(ILDs).item()



def recall_precision_at_k(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])

    recall = np.sum(right_pred / recall_n)
    precision = np.sum(right_pred) / k

    return {"recall": recall, "precision": precision}


def mrr_at_k_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, : k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)

    return np.sum(pred_data)


def ndcg_at_k_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)

    pred_data = r[:, : k]
    test_matrix = np.zeros((len(pred_data), k))

    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, : length] = 1

    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.

    return np.sum(ndcg)


def auc(all_item_scores, dataset, test_data):
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]

    return roc_auc_score(r, test_item_scores)


def get_label(test_data, pred_data):
    r = []

    for i in range(len(test_data)):
        ground_true = test_data[i]
        predict_top_k = pred_data[i]

        pred = list(map(lambda x: x in ground_true, predict_top_k))
        pred = np.array(pred).astype("float")

        r.append(pred)

    return np.array(r).astype('float')
