"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
"""
import utils
import torch
import world

from utils import Timer
from tqdm import tqdm


def train_pairwise(dataset, model, loss_class, optimizer):
    """
    Train the model using pairwise ranking loss.

    Args:
        dataset (BasicDataset): The training dataset.
        model (BasicModel): The model to train.
        loss_class: The loss function class.
        optimizer: The optimizer for model parameters.

    Returns:
        tuple: A tuple containing the average loss value and timing info.
    """
    model.train()

    # uniform sample for bpr
    with Timer(name="Sample"):
        samples = utils.uniform_sample_original(dataset)

    users = torch.Tensor(samples[:, 0]).long()
    pos_items = torch.Tensor(samples[:, 1]).long()
    neg_items = torch.Tensor(samples[:, 2]).long()

    users = users.to(world.device)
    pos_items = pos_items.to(world.device)
    neg_items = neg_items.to(world.device)

    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)
    total_batch = len(users) // world.config["batch_size"] + 1
    avg_loss = 0.

    for (batch_i, (batch_users, batch_pos, batch_neg)) \
            in tqdm(enumerate(utils.minibatch(users, pos_items, neg_items,
                    batch_size=world.config["batch_size"])), desc="Training",
                    total=total_batch, leave=False):
        optimizer.zero_grad()
        all_users_embeddings, all_items_embeddings = model()

        users_embeddings = all_users_embeddings[batch_users]
        pos_items_embeddings = all_items_embeddings[batch_pos]
        neg_items_embeddings = all_items_embeddings[batch_neg]

        users_embeddings_layer0 = model.embedding_user(batch_users)
        pos_items_embeddings_layer0 = model.embedding_item(batch_pos)
        neg_items_embeddings_layer0 = model.embedding_item(batch_neg)

        loss = loss_class(
            users_embeddings,
            pos_items_embeddings,
            neg_items_embeddings,
            users_embeddings_layer0,
            pos_items_embeddings_layer0,
            neg_items_embeddings_layer0,
            model.parameters_norm()
        )

        loss.backward()
        optimizer.step()

        avg_loss += loss.item()

    avg_loss = avg_loss / total_batch
    time_info = Timer.dict()
    Timer.zero()

    return avg_loss, time_info
