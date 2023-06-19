"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
"""
import multiprocessing
import utils
import world
import numpy as np
import torch

from datasets import BasicDataset
from models import BasicModel
from tqdm import tqdm


def test_one_batch(X):
    """
    Calculate precision, recall, and NDCG for a batch of user-item pairs.

    Args:
        X (tuple): A tuple containing sorted items and ground truth for a batch of user-item pairs.

    Returns:
        dict: Dictionary containing recall, precision, and NDCG values for different top-k recommendations.
    """
    sorted_items = X[0].numpy()
    ground_truth = X[1]

    label = utils.get_label(ground_truth, sorted_items)
    precision, recall, ndcg = [], [], []

    for k in world.topks:
        ret = utils.recall_precision_at_k(ground_truth, label, k)
        precision.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.ndcg_at_k_r(ground_truth, label, k))

    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg)
    }


def eval_pairwise(dataset: BasicDataset, model: BasicModel, multicore=0):
    """
    Evaluate the pairwise ranking performance of the model on the test dataset.

    Args:
        dataset (BasicDataset): The test dataset.
        model (BasicModel): The trained model.
        multicore (int): Number of CPU cores to use for parallel processing.

    Returns:
        dict: Dictionary containing recall, precision, and NDCG values for different top-k recommendations.
    """
    batch_size = world.config["test_u_batch_size"]
    test_dict = dataset.test_dict

    model = model.eval()
    max_k = max(world.topks)

    if multicore:
        pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)

    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks))
    }

    with torch.no_grad():
        users = list(test_dict.keys())

        users_list = []
        rating_list = []
        ground_truth_list = []

        total_batch = len(users) // batch_size + 1
        for batch_users in tqdm(utils.minibatch(users, batch_size=batch_size),
                                desc="Validation",
                                total=total_batch,
                                leave=False):
            ground_truth = [test_dict[user] for user in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)

            rating = model.get_user_rating(batch_users_gpu)
            all_pos = dataset.get_user_pos_items(batch_users)

            exclude_index = []
            exclude_items = []

            for range_i, items in enumerate(all_pos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)

            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_k)
            rating = rating.cpu().numpy()

            del rating

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            ground_truth_list.append(ground_truth)

        assert total_batch == len(users_list)

        X = zip(rating_list, ground_truth_list)

        if multicore:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]

        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))

        if multicore:
            pool.close()

        return results
