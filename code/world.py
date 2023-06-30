"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al.
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
"""

import os
import torch
import sys
import multiprocessing

from parse import parse_args
from parse import all_datasets
from parse import all_models
from os.path import join
from warnings import simplefilter


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
args = parse_args()

WANDB_PROJECT = "recsys"
WANDB_ENTITY = "msc-ai"

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, "code")
DATA_PATH = join(ROOT_PATH, "data")
BOARD_PATH = join(CODE_PATH, "runs")
EMBS_PATH = join(CODE_PATH, "embs")
FILE_PATH = join(CODE_PATH, "checkpoints")
sys.path.append(join(CODE_PATH, "sources"))

for path_name in [FILE_PATH, EMBS_PATH]:
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)

config = {}

config["batch_size"] = args.batch_size
config["latent_dim_rec"] = args.recdim
config["lightGCN_n_layers"] = args.layer
config["dropout"] = args.dropout
config["keep_prob"] = args.keepprob
config["adj_matrix_folds"] = args.a_fold
config["test_u_batch_size"] = args.testbatch
config["multicore"] = args.multicore
config["lr"] = args.lr
config["decay"] = args.decay
config["pretrain"] = args.pretrain
config["A_split"] = False
config["bigdata"] = False
config["seed"] = args.seed
config["topks"] = eval(args.topks)
config["single"] = args.single
config["l1"] = args.l1
config["side_norm"] = args.side_norm
config["embs_path"] = EMBS_PATH
config["save_embs"] = args.save_embs
config["dataset"] = args.dataset
config["model"] = args.model
config["save_model_by"] = args.save_model_by

# Attention
if "attention_dim" in args and config["model"] == "w-sdp-a-lgn":
    config["attention_dim"] = args.attention_dim

# APPNP
if 'num_walks' in args and 'alpha' in args and config['model'] == 'appnp':
    config["num_walks"] = args.num_walks
    config["alpha"] = args.alpha

GPU = torch.cuda.is_available()
device = torch.device("cuda" if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = config["dataset"]
model_name = config["model"]
if config["dataset"] not in all_datasets:
    raise NotImplementedError(
        f"Haven't supported {config['dataset']} yet!, try {all_datasets}")
if config["model"] not in all_models:
    raise NotImplementedError(
        f"Haven't supported {config['model']} yet!, try {all_models}")

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
comment = args.comment
num_bins = 20

# let pandas shut up
simplefilter(action="ignore", category=FutureWarning)

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""

# font: ANSI Shadow
# refer to
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
