"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import numpy as np
import world

from cppimport import imp_from_filepath
from os.path import join


try:
    path = join("sources", "sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except Exception:
    print("Cpp extension not loaded")
    sample_ext = False


def uniform_sample_original(dataset, neg_ratio=1):
    """
    Uniformly sample negative items for BPR training.

    Args:
        dataset (BasicDataset): The dataset object.
        neg_ratio (int, optional): The negative sampling ratio. Defaults to 1.

    Returns:
        np.array: Array of sampled negative items.
    """
    all_pos = dataset.all_pos

    if sample_ext:
        samples = sampling.sample_negative(
            dataset.n_users, dataset.m_items,
            dataset.train_data_size, all_pos, neg_ratio
        )
    else:
        samples = uniform_sample_original_python(dataset)

    return samples


def uniform_sample_original_python(dataset):
    """
    The original implementation of BPR Sampling in LightGCN.

    Args:
        dataset (BasicDataset): The dataset object.

    Returns:
        np.array: Array of sampled negative items.
    """
    user_num = dataset.train_data_size
    users = np.random.randint(0, dataset.n_users, user_num)
    all_pos = dataset.all_pos
    samples = []

    for user in users:
        pos_for_user = all_pos[user]

        if len(pos_for_user) == 0:
            continue

        pos_index = np.random.randint(0, len(pos_for_user))
        pos_item = pos_for_user[pos_index]

        while True:
            neg_item = np.random.randint(0, dataset.m_items)
            if neg_item in pos_for_user:
                continue
            else:
                break

        samples.append([user, pos_item, neg_item])

    return np.array(samples)


def minibatch(*tensors, **kwargs):
    """
    Create minibatches from input tensors.

    Args:
        tensors (tuple): Tuple of input tensors.
        batch_size (int, optional): The batch size. Defaults to the value in
                                    world.config.

    Yields:
        tuple: Tuple of minibatches.
    """
    batch_size = kwargs.get("batch_size", world.config["batch_size"])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i: i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i: i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """
    Shuffle input arrays.

    Args:
        arrays (tuple): Tuple of input arrays.
        indices (bool, optional): Whether to return the shuffled indices.
                                  Defaults to False.

    Returns:
        tuple or np.array: Shuffled arrays or tuple of shuffled arrays and
                           shuffled indices.
    """
    require_indices = kwargs.get("indices", False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("All inputs to shuffle must have the same length.")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result
