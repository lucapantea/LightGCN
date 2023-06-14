import world
import utils
import register
import torch
import wandb
import time
import Procedure
import os

from os.path import join
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from register import dataset

def main():
    # Set seed
    utils.set_seed(world.seed)

    # Initialize model
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)

    # Initialize BPR loss
    bpr = utils.BPRLoss(Recmodel, world.config)

    # Load pretrain weights
    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")

    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # Creating the run name
    num_layers = world.config['lightGCN_n_layers']
    run_name = f"{world.model_name}_{world.dataset}" \
               f"{f'_layers-{num_layers}' if world.model_name not in ['mf'] else ''}" \
               f"_latent_dim-{world.config['latent_dim_rec']}"

    # Initialize tensorboard and wandb
    tensorboard_log_dir = join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + run_name)
    wandb.tensorboard.patch(root_logdir=tensorboard_log_dir)
    wandb.init(project="recsys", entity="msc-ai", config=world.config,
               reinit=True, name=run_name)
    wandb.watch(Recmodel)

    # Initialize tensorboard writer
    if world.tensorboard:
        w: SummaryWriter = SummaryWriter(tensorboard_log_dir)
    else:
        w = None
        world.cprint("Tensorboard not enabled.")

    # Saving the best model instance based on set variable
    save_model_by = 'ndcg'  # metrics: precision, recall, ndcg
    best_test_metric = float('-inf')

    try:
        print("Beginning training...")
        with tqdm(range(world.TRAIN_epochs), desc="Epoch") as pbar:
            for epoch in pbar:
                avg_loss, sampling_time = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
                wandb.log({"BPR Loss": avg_loss, "Epoch": epoch})

                # Evaluate the model on the validation set
                if epoch % 10 == 0:
                    test_metrics = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                    wandb.log({**test_metrics, 'Epoch': epoch})

                    if test_metrics[save_model_by] > best_test_metric:
                        best_test_metric = test_metrics[save_model_by]
                        wandb.run.summary[f"best_{save_model_by}"] = best_test_metric
                        ckpt = {"state_dict": Recmodel.state_dict(),
                                "optimizer_state_dict": bpr.opt.state_dict(),
                                f"best_{save_model_by}": best_test_metric,
                                "best_epoch": epoch}
                        torch.save(ckpt, weight_file)
                        if world.config['save_embs']:
                            for i, emb in enumerate(Recmodel.get_embedding_matrix().unbind(dim=1)):
                                torch.save(emb, os.path.join(world.config['embs_path'],
                                                             f"emb_layer-{i}_{os.path.basename(weight_file)}"))

                # Update the progress bar
                pbar.set_postfix({'BPR loss': f'{avg_loss:.3f}', 'sampling time': sampling_time})

    except KeyboardInterrupt:
        # Training can be safely interrupted with Ctrl+C
        print('Exiting training early because of keyboard interrupt.')
    finally:
        wandb.finish()
        if world.tensorboard:
            w.close()


if __name__ == '__main__':
    main()
