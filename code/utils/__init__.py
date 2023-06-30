from .metrics import *
from .samplings import *
from .Timer import Timer
from .optimization import create_optimizer, create_scheduler
from datasets import BasicDataset, LastFM, Loader

import os
import numpy as np
import torch
import world


def set_seed(seed: int):
    """
    Set the seed for random number generators for reproducibility.

    Args:
        seed (int): The seed value.

    Returns:
        None
    """
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def get_weights_file_name(checkpoint_path, model_name, num_layers, single, l1,
                          side_norm_type, dataset, latent_dim_rec, batch_size,
                          dropout, keep_prob, adj_matrix_folds,
                          test_u_batch_size, lr, decay, seed, **kwargs):
    """
    Generate the file name for saving weights of a model.

    Args:
        checkpoint_path (str): The path to the checkpoint directory.
        model_name (str): The name of the model.
        num_layers (int): The number of layers.
        single (bool): Whether to use single layer.
        l1 (bool): Whether to use L1 regularization.
        side_norm_type (str): The type of side information normalization.
        dataset (str): The dataset name.
        latent_dim_rec (int): The dimension of the latent space.
        batch_size (int): The batch size for training.
        dropout (float): The dropout rate.
        keep_prob (float): The keep probability for dropout.
        adj_matrix_folds (int): The number of folds for splitting the
                                adjacency matrix.
        test_u_batch_size (int): The batch size for testing.
        lr (float): The learning rate.
        decay (float): The weight decay.
        seed (int): The seed value.
        **kwargs: Additional keyword arguments for other parameters.

    Returns:
        str: The file path for saving the weights.
    """
    single = "-single" if single else ""
    l1 = "-L1" if l1 else ""

    if side_norm_type.upper() in ["L", "R"]:
        side_norm = f"-{side_norm_type.upper()}"
    else:
        side_norm = ""

    file_name = f"{model_name}{single}{l1}{side_norm}_{dataset}" \
        f"{f'_layers-{num_layers}' if 'lgn' in model_name.lower() else ''}" \
        f"_latent_dim-{latent_dim_rec}" \
        f"_bpr_batch_size-{batch_size}" \
        f"_dropout-{dropout}" \
        f"_keep_prob-{keep_prob}" \
        f"_A_n_fold-{adj_matrix_folds}" \
        f"_test_u_batch_size-{test_u_batch_size}" \
        f"_lr-{lr}" \
        f"_decay-{decay}" \
        f"_seed-{seed}"

    # Append additional parameters
    for key, value in kwargs.items():
        file_name += f"_{key}-{value}"

    file_name += ".pt"

    return os.path.join(checkpoint_path, file_name)


def get_wandb_run_name(model_name, dataset, num_layers, latent_dim, **kwargs):
    """
    Generate the run name for Weights & Biases logging.

    Args:
        model_name (str): The name of the model.
        dataset (str): The dataset name.
        num_layers (int): The number of layers.
        latent_dim (int): The dimension of the latent space.
        **kwargs: Additional keyword arguments for other parameters.

    Returns:
        str: The run name for Weights & Biases logging.
    """
    use_layers = f"_layers-{num_layers}" if world.model_name != "mf" else ""
    wandb_run_name = f"{model_name}_{dataset}" \
                     f"{use_layers}" \
                     f"_latent_dim-{latent_dim}"

    # Append additional parameters
    for key, value in kwargs.items():
        wandb_run_name += f"_{key}-{value}"

    return wandb_run_name


def get_dataset(data_path: str, dataset: BasicDataset):
    if dataset in ["gowalla", "yelp2018", "amazon-book", "citeulike", "movielens", "amazon-beauty", "amazon-cds", "amazon-electro", "amazon-movies"]:
        return Loader(
            config=world.config,
            path=os.path.join(data_path, dataset))
    elif dataset == "lastfm":
        return LastFM()
