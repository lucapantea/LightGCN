import world
import utils
import torch
import wandb
import os
import procedures
import numpy as np

from tqdm import tqdm
from pprint import pprint
from models import LightGCN
from models import PureMF
from models import BaseAttention
from models import FinerAttention
from models import ScaledDotProductAttentionLightGCN
from models import WeightedScaledDotProductAttentionLightGCN
from losses import BPRLoss


MODELS = {
    "mf": PureMF,
    "lgn": LightGCN,
    "base-a-lgn": BaseAttention,
    "finer-a-lgn": FinerAttention,
    "sdp-a-lgn": ScaledDotProductAttentionLightGCN,
    "w-sdp-a-lgn": WeightedScaledDotProductAttentionLightGCN,
    "appnp": LightGCN
}


def main():
    # Set seed
    utils.set_seed(world.seed)

    if world.model_name not in MODELS:
        raise NotImplementedError(
            f"Model name '{world.model_name}' not recognized.")

    # Initialize model
    dataset = utils.get_dataset(world.DATA_PATH, world.dataset)
    model = MODELS[world.model_name](world.config, dataset)
    model = model.to(world.device)

    # Initialize BPR loss
    criterion = BPRLoss(weight_decay=world.config["decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=world.config["lr"])

    print("===========config================")
    pprint(world.config)
    print("cores for test:", world.CORES)
    print("comment:", world.comment)
    print("LOAD:", world.LOAD)
    print("Weight path:", world.PATH)
    print("Test Topks:", world.topks)
    print("Loss function:", criterion.__class__)
    print("===========end===================")

    using_attention = "attention_dim" in world.config
    using_scaled_dot_prod = world.config["model"] == "w-sdp-a-lgn"

    # Get the checkpoint filename
    weight_file = utils.get_weights_file_name(
        checkpoint_path=world.FILE_PATH,
        model_name=world.model_name,
        num_layers=world.config["lightGCN_n_layers"],
        single=world.config["single"],
        l1=world.config["l1"],
        side_norm_type=world.config["side_norm"],
        dataset=world.dataset,
        latent_dim_rec=world.config["latent_dim_rec"],
        batch_size=world.config["batch_size"],
        dropout=world.config["dropout"],
        keep_prob=world.config["keep_prob"],
        adj_matrix_folds=world.config["adj_matrix_folds"],
        test_u_batch_size=world.config["test_u_batch_size"],
        lr=world.config["lr"],
        decay=world.config["decay"],
        seed=world.config["seed"],
        # Extra weight filename parameters are now supported.

        # Weighted scaled dot product
        **{"attention_dim": world.config["attention_dim"]} if (
            using_attention and using_scaled_dot_prod
        ) else {}
    )
    print(f"Loading and saving to {weight_file}")

    if world.LOAD:
        try:
            model.load_state_dict(torch.load(
                weight_file, map_location=world.device))
            print(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")

    # Creating the run name
    wandb_run_name = utils.get_wandb_run_name(
        model_name=world.model_name,
        dataset=world.dataset,
        num_layers=world.config["lightGCN_n_layers"],
        latent_dim=world.config["latent_dim_rec"]
        # Extra wandb filename parameters are now supported.
    )

    # Initialize wandb
    wandb.init(project=world.WANDB_PROJECT, entity=world.WANDB_ENTITY,
               config=world.config, reinit=True, name=wandb_run_name,
               tags=["latest"])
    wandb.watch(model)

    # Saving the best model instance based on set variable
    save_model_by = world.config["save_model_by"]
    best_test_metric = float("-inf")

    bins = [f"Bin {bin_num + 1}" for bin_num in range(world.num_bins)]
    try:
        print("Beginning training...")
        with tqdm(range(world.TRAIN_epochs), desc="Epoch") as pbar:
            for epoch in pbar:
                avg_loss, sampling_time = procedures.train_pairwise(
                    dataset, model, criterion, optimizer)
                wandb.log({"BPR Loss": avg_loss, "Epoch": epoch})

                # Evaluate the model on the validation set
                if epoch % 10 == 0:
                    test_metrics = procedures.eval_pairwise(
                        dataset, model, world.config["multicore"])

                    # Result dictionary to log to wandb
                    results = {}

                    # Exploration table: Metrics vs. Bin
                    exploration_table_data = []
                    columns = ["Metric"] + bins

                    # For each k, log the precision, recall, and ndcg,
                    # and construct the exploration table
                    for i_k, k in enumerate(world.topks):
                        results[f"Precision@{k}"] = test_metrics["precision"][i_k]
                        results[f"Recall@{k}"] = test_metrics["recall"][i_k]
                        results[f"NDCG@{k}"] = test_metrics["ndcg"][i_k]
                        exploration_table_data.append(
                            [f"Exploration_vs_precision@{k}"] + test_metrics["exploration_vs_precision"][i_k, :].tolist())
                        exploration_table_data.append(
                            [f"Exploration_vs_recall@{k}"] + test_metrics["exploration_vs_recall"][i_k, :].tolist())
                        exploration_table_data.append(
                            [f"Exploration_vs_ndcg@{k}"] + test_metrics["exploration_vs_ndcg"][i_k, :].tolist())

                    # Log the exploration table, diversity, and novelty
                    results["Exploration Table"] = wandb.Table(
                        data=exploration_table_data, columns=columns)
                    results["Diversity"] = test_metrics["diversity"]
                    results["Novelty"] = test_metrics["novelty"]

                    # Log the results to wandb
                    wandb.log({**results, "Epoch": epoch})

                    # Save the model if it is the best so far
                    best_val = np.argmax(test_metrics[save_model_by])
                    if test_metrics[save_model_by][best_val] > best_test_metric:
                        best_test_metric = test_metrics[save_model_by][best_val]
                        run_name = f"best_{save_model_by}"
                        wandb.run.summary[run_name] = best_test_metric
                        ckpt = {
                            "state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            f"{run_name}@{max(world.topks)}": best_test_metric,
                            "best_epoch": epoch
                        }
                        torch.save(ckpt, weight_file)
                        if world.config["save_embs"]:
                            for i, emb in enumerate(model.get_embedding_matrix().unbind(dim=1)):
                                name_weight_file = os.path.basename(weight_file)
                                torch.save(
                                    emb,
                                    os.path.join(
                                        world.config["embs_path"],
                                        f"emb_layer-{i}_{name_weight_file}")
                                )

                # Update the progress bar
                pbar.set_postfix({
                    "BPR loss": f"{avg_loss:.3f}",
                    "sampling time": sampling_time
                })

    except KeyboardInterrupt:
        # Training can be safely interrupted with Ctrl+C
        print("Exiting training early because of keyboard interrupt.")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
