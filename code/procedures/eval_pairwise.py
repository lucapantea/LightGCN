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


def test_one_batch(X, item_embeddings, conversion_interaction_to_bin, batch_n, train_items_interacted_batch, num_bins=10):
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
    exploration_vs_precision = np.zeros((len(world.topks), num_bins))
    exploration_vs_recall = np.zeros((len(world.topks), num_bins))
    exploration_vs_ndcg = np.zeros((len(world.topks), num_bins))

    for i_k, k in enumerate(world.topks):
        ret = utils.recall_precision_at_k(ground_truth, label, k)
        precision.append(ret["precision"])
        recall.append(ret["recall"])
        ndcg.append(utils.ndcg_at_k_r(ground_truth, label, k))

        for user in range(len(sorted_items)):
            n_interactions = len(ground_truth[user])
            user_bin = conversion_interaction_to_bin[n_interactions]
            user_ret = utils.recall_precision_at_k(ground_truth[user], label[user,:], k)
            user_ndcg = utils.ndcg_at_k_r(ground_truth[user], label[user], k)
            user_precision = user_ret['precision']
            user_recall = user_ret['recall']
            exploration_vs_precision[i_k, user_bin] += user_precision
            exploration_vs_recall[i_k, user_bin] += user_recall
            exploration_vs_ndcg[i_k, user_bin] = user_ndcg

    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "diversity": utils.mean_intra_list_distance(recommendation_lists=sorted_items,
                                                    item_embeddings=item_embeddings),
        'novelty': utils.novelty(ground_truth[batch_n*100:(batch_n+1)*100], train_items_interacted_batch[batch_n], 20),
        'exploration_vs_precision': exploration_vs_precision,
        'exploration_vs_recall': exploration_vs_recall,
        'exploration_vs_ndcg': exploration_vs_ndcg
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

    # Data for exploration_vs_recall metric
    number_of_train_interactions = np.array(dataset.number_of_train_interactions)
    # Define the bin thresholds
    num_bins = 10
    bin_thresholds = np.linspace(number_of_train_interactions.min(), number_of_train_interactions.max(), num_bins)

    # Assign users to bins using np.digitize()
    bin_indices = np.digitize(number_of_train_interactions, bin_thresholds)
    conversion_interaction_to_bin = dict(zip(number_of_train_interactions, bin_indices))
    bin_counts = np.bincount(bin_indices)

    # Data for diversity
    train_items_interacted = np.array(dataset.train_items_interacted)
    # Define the number of rows for each mini matrix
    rows_per_mini_matrix = world.config['test_batch']
    # Split the matrix into mini matrices
    batch_matrices = np.array_split(train_items_interacted, train_items_interacted.shape[0] // rows_per_mini_matrix)
    # Create a dictionary to store the mini matrices
    train_items_interacted_batch = {i: batch_matrices[i] for i in range(len(batch_matrices))}


    results = {
            "precision": np.zeros(len(world.topks)),
            "recall": np.zeros(len(world.topks)),
            "ndcg": np.zeros(len(world.topks)),
            "diversity": 0.,
            'novelty': 0.,
            'exploration_vs_precision': np.zeros((len(world.topks), num_bins)),
            "exploration_vs_recall": np.zeros((len(world.topks), num_bins)),
            'exploration_vs_ndcg': np.zeros((len(world.topks), num_bins))
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

        # Perform forward pass of the model to obtain the item embeddings
        _, item_embeddings = model()

        # if multicore:
        #     pre_results = pool.map(test_one_batch, X, item_embeddings)
        # else:
        pre_results = []
        for batch_n, x in enumerate(X):
            pre_results.append(test_one_batch(x, item_embeddings, conversion_interaction_to_bin, batch_n, train_items_interacted_batch, num_bins=num_bins))

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results['diversity'] += result["diversity"]
            results['novelty'] += result["novelty"]
            results['exploration_vs_precision'] += result['exploration_vs_precision']
            results['exploration_vs_recall'] += result['exploration_vs_recall']
            results['exploration_vs_ndcg'] += result['exploration_vs_ndcg']

        results["recall"] /= float(len(users))
        results["precision"] /= float(len(users))
        results["ndcg"] /= float(len(users))
        results["diversity"] /= float(len(users))
        results["novelty"] /= float(len(users))
        results['exploration_vs_precision'] /= bin_counts
        results['exploration_vs_recall'] /= bin_counts
        results['exploration_vs_ndcg'] /= bin_counts

        if multicore:
            pool.close()

        return results
