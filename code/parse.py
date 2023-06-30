"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""
import json
import argparse

all_datasets = ["lastfm", "gowalla", "yelp2018", "amazon-book", "citeulike", "movielens", "amazon-beauty", "amazon-cds", "amazon-electro", "amazon-movies"]
all_models = ["mf", "lgn", "base-a-lgn", "finer-a-lgn", "sdp-a-lgn", "w-sdp-a-lgn", "appnp"]


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument("--recdim", type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument("--layer", type=int, default=3,
                        help="the layer num of lightGCN")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument("--decay", type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument("--dropout", type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument("--keepprob", type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument("--a_fold", type=int, default=100,
                        help="the fold num used to split large adj matrix")
    parser.add_argument("--testbatch", type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument("--dataset", type=str, default="gowalla",
                        help=f"available datasets: {str(all_datasets)}")
    parser.add_argument("--path", type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument("--topks", nargs="?", default="[1, 2, 3, 5, 10, 20]",
                        help="@k test list")
    parser.add_argument("--comment", type=str, default="lgn")
    parser.add_argument("--load", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--multicore", type=int, default=0,
                        help="whether to use multiprocessing or not in test")
    parser.add_argument("--pretrain", type=int, default=0,
                        help="whether to use pretrained weight or not")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--model", type=str, default="lgn",
                        help=f"rec-model, support {str(all_models)}")
    parser.add_argument("--single", action="store_true", default=False,
                        help="whether to use single LightGCN to test")
    parser.add_argument("--save_embs", action="store_true", default=False,
                        help="whether or not to store the embedding matrices")
    parser.add_argument("--l1", action="store_true", default=False,
                        help="whether we use L1 norm for adj matrix")
    parser.add_argument("--side_norm", type=str, default="both",
                        help="available norms: [l, r, both]")
    parser.add_argument("--save_model_by", type=str, default="ndcg",
                        help="available metrics: [ndcg, recall, precision]")

    # Optimization
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="optimizer to use for training")
    parser.add_argument("--scheduler", type=str, default="step_lr",
                        help="scheduler to use for adjusting learning rate")
    parser.add_argument("--scheduler_params", type=json.loads, default={},
                        help="more params for the scheduler in JSON format")

    # Attention
    parser.add_argument("--attention_dim", type=int, default=2,
                        help="Number of dims for the attention projections")

    # APPNP
    parser.add_argument('--num_walks', type=int, default=10,
                        help='Number of random walk steps for APPNP')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='The teleportation parameter for APPNP')

    return parser.parse_args()
