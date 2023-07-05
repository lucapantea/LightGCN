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

from tqdm import tqdm


def test_one_batch(X, item_embeddings, batch_user_bins,
                   batch_user_interaction_history, num_bins=20):
    """
    Evaluate the performance of a model on a batch of test data.

    Args:
        X (tuple): A tuple containing the predicted ratings and ground truth
                  labels for a batch of test data.
        item_embeddings (torch.Tensor): Embeddings of items.
        batch_user_bins (list): List of user bins indicating the number of
                                interactions.
        batch_user_interaction_history (list): List of user interaction
                                               histories.
        num_bins (int, optional): Number of bins for binning user interactions.
                                  Defaults to 20.

    Returns:
        dict: A dictionary containing evaluation metrics.
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
            user_bin = batch_user_bins[user]

            user_ground_truth = np.expand_dims(ground_truth[user], axis=0)
            user_label = np.expand_dims(label[user, :], axis=0)

            user_ret = utils.recall_precision_at_k(
                user_ground_truth, user_label, k)
            user_ndcg = utils.ndcg_at_k_r(
                user_ground_truth, user_label, k)
            user_precision = user_ret["precision"]
            user_recall = user_ret["recall"]

            # Binning by number of interactions
            exploration_vs_precision[i_k, user_bin] += user_precision
            exploration_vs_recall[i_k, user_bin] += user_recall
            exploration_vs_ndcg[i_k, user_bin] = user_ndcg

    return {
        "recall": np.array(recall),
        "precision": np.array(precision),
        "ndcg": np.array(ndcg),
        "diversity": utils.mean_intra_list_distance(
            recommendation_lists=sorted_items,
            item_embeddings=item_embeddings
        ),
        "novelty": utils.novelty(
            ground_truth, batch_user_interaction_history,
            max(world.topks)
        ),
        "exploration_vs_precision": exploration_vs_precision,
        "exploration_vs_recall": exploration_vs_recall,
        "exploration_vs_ndcg": exploration_vs_ndcg
    }


def eval_pairwise(dataset, model, multicore=0):
    """
    Evaluate the pairwise ranking model on the test data.

    Args:
        dataset (BasicDataset): The dataset containing user-item interactions.
        model (BasicModel): The pairwise ranking model.
        multicore (int, optional): Number of cores to use for parallel 
                                   processing. Defaults to 0.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    batch_size = world.config["test_u_batch_size"]
    test_dict = dataset.test_dict

    model = model.eval()
    max_k = max(world.topks)

    if multicore:
        multiprocessing.set_start_method("spawn", force=True)
        pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)

    # Define the bin thresholds
    num_bins = world.num_bins

    results = {
        "precision": np.zeros(len(world.topks)),
        "recall": np.zeros(len(world.topks)),
        "ndcg": np.zeros(len(world.topks)),
        "diversity": 0.,
        "novelty": 0.,
        "exploration_vs_precision": np.zeros((len(world.topks), num_bins)),
        "exploration_vs_recall": np.zeros((len(world.topks), num_bins)),
        "exploration_vs_ndcg": np.zeros((len(world.topks), num_bins))
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

        # The user bins are computed by binning users based on the number of
        # interactions they have in the training set
        user_bins_by_num_interactions = [
            [dataset.user_bins_by_num_interactions[user_id]
             for user_id in batch_users] for batch_users in users_list
        ]

        user_interaction_history = [
            [dataset.user_interactions_dict_train[user_id]
             for user_id in batch_users] for batch_users in users_list
        ]

        if multicore:
            pre_results = pool.starmap(
                test_one_batch,
                [(x, item_embeddings, user_bins_by_num_interactions[batch],
                 user_interaction_history[batch], num_bins)
                 for batch, x in enumerate(X)]
            )
        else:
            pre_results = []
            for batch, x in enumerate(X):
                pre_results.append(
                    test_one_batch(
                        x, item_embeddings,
                        user_bins_by_num_interactions[batch],
                        user_interaction_history[batch], num_bins
                    )
                )

        for result in pre_results:
            results["recall"] += result["recall"]
            results["precision"] += result["precision"]
            results["ndcg"] += result["ndcg"]
            results["diversity"] += result["diversity"]
            results["novelty"] += result["novelty"]
            results["exploration_vs_precision"] += \
                result["exploration_vs_precision"]
            results["exploration_vs_recall"] += result["exploration_vs_recall"]

    return results
