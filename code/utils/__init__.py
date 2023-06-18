from .metrics import *
from .samplings import *
from .Timer import Timer
from datasets import BasicDataset, LastFM, Loader

import os
import numpy as np
import torch
import world


def set_seed(seed: int):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def get_weights_file_name(
    checkpoint_path,
    model_name: str,
    num_layers: int,
    single: bool,
    l1: bool,
    side_norm_type: str,
    dataset: str,
    latent_dim_rec: int,
    batch_size: int,
    dropout: int,
    keep_prob: float,
    adj_matrix_folds: int,
    test_u_batch_size: int,
    lr: float,
    decay: float,
    seed: int,
    **kwargs
):
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

def get_wandb_run_name(model_name, dataset, num_layers, latent_dim_rec, **kwargs):
    use_layers = f"_layers-{num_layers}" if world.model_name != "mf" else ""
    wandb_run_name = f"{model_name}_{dataset}" \
                     f"{use_layers}" \
                     f"_latent_dim-{latent_dim_rec}"

    # Append additional parameters
    for key, value in kwargs.items():
        wandb_run_name += f"_{key}-{value}"

    return wandb_run_name


def get_dataset(data_path: str, dataset: BasicDataset):
    if dataset in ["gowalla", "yelp2018", "amazon-book"]:
        return Loader(
            config=world.config,
            path=os.path.join(data_path, dataset))
    elif dataset == "lastfm":
        return LastFM()
