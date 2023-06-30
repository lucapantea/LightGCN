import torch
from datasets import Loader
import os
import world
import numpy as np
import math
from tqdm import tqdm


def compute_smoothness(graph, embeddings, type_, num_users):
    """
    Compute the smoothness of embeddings based on the given graph.

    Args:
        graph (list): The graph representing the user-item interactions.
        embeddings (torch.Tensor): Embeddings of users and items.
        type_ (str): The type of embeddings ("users" or "items").
        num_users (int): Number of users.

    Returns:
        float: The smoothness of the embeddings.
    """
    smoothness = 0

    idx_start = 0 if type_ == "users" else num_users
    idx_end = num_users if type_ == "users" else len(embeddings)

    for u in tqdm(range(idx_start, idx_end)):
        for v in tqdm(range(idx_start, idx_end), leave=False):
            eu = embeddings[u]
            ev = embeddings[v]

            neighbors_u = graph[u].coalesce().indices().squeeze()
            neighbors_v = graph[v].coalesce().indices().squeeze()

            Nu = math.sqrt(neighbors_u.shape[0])
            Nv = math.sqrt(neighbors_v.shape[0])

            smoothness_strength = 0

            for i in np.intersect1d(neighbors_u.cpu(), neighbors_v.cpu()):
                neighbors = graph[i].coalesce().indices().squeeze()
                num_neighbors = neighbors.shape[0]
                smoothness_strength += 1 / num_neighbors

            smoothness_strength /= (Nu * Nv)
            smoothness += smoothness_strength * (eu - ev).pow(2).sum()

    return smoothness


if __name__ == "__main__":
    for emb_file in os.listdir(world.EMBS_PATH):
        _, layer, method, dataset = emb_file.split("_")[: 4]

        world.dataset = dataset
        dataloader = Loader(
            config=world.config, path=os.path.join("..", "data", dataset)
        )
        graph = dataloader.get_sparse_graph().to(world.device)
        num_users = dataloader.n_user

        emb_file_path = os.path.join(world.EMBS_PATH, emb_file)
        embeddings = torch.load(emb_file_path).to(world.device)
        embeddings /= torch.linalg.norm(embeddings, dim=1, ord=2).unsqueeze(-1)

        users_smoothness = compute_smoothness(
            graph, embeddings, type_="users", num_users=num_users)
        items_smoothness = compute_smoothness(
            graph, embeddings, type_="items", num_users=num_users)

        with open("smoothness_results.txt", "a") as w:
            w.write("users")
            w.write(" ")
            w.write(method)
            w.write(" ")
            w.write(dataset)
            w.write(" ")
            w.write(str(round(users_smoothness.item(), 2)))
            w.write("\n")
            w.write("items")
            w.write(" ")
            w.write(method)
            w.write(" ")
            w.write(dataset)
            w.write(" ")
            w.write(str(round(items_smoothness.item(), 2)))
            w.write("\n")
