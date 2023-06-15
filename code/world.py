'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
import torch
import sys
from parse import parse_args
from os.path import join
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
EMBS_PATH = join(CODE_PATH, 'embs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
sys.path.append(join(CODE_PATH, 'sources'))

def prepare_dir(path_name):
    """
    This function is used to create the directories needed to output a path. If the directories already exist, the
    function continues.
    """
    # Try to create the directory. Will have no effect if the directory already exists.
    if not os.path.exists(path_name):
        os.makedirs(path_name, exist_ok=True)

for path_name in [FILE_PATH, EMBS_PATH]:
    prepare_dir(path_name)

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models = ['mf', 'lgn', 'attention-lgn', 'finer-attention-lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers'] = args.layer
config['dropout'] = args.dropout
config['keep_prob'] = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['seed'] = args.seed
config['topks'] = eval(args.topks)
config['single'] = args.single
config['l1'] = args.l1
config['side_norm'] = args.side_norm
config['embs_path'] = EMBS_PATH
config['save_embs'] = args.save_embs
config['dataset'] = args.dataset
config['model'] = args.model

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

if config['dataset'] not in all_dataset:
    raise NotImplementedError(f"Haven't supported {config['dataset']} yet!, try {all_dataset}")
if config['model'] not in all_models:
    raise NotImplementedError(f"Haven't supported {config['model']} yet!, try {all_models}")

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
