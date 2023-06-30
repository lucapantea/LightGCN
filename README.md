
## LightGCN-pytorch

This is our Pytorch implementation adaptation for our reproducibility case study of [LightGCN](https://arxiv.org/abs/2002.02126):

## Introduction

In this work, we aim to not only reproduce the LightGCN paper, but also close the research gap by coming up with additional experiments and evaluation methods to see how different variants of LightGCN behave in different scenarios.

## Getting started

In order to set up the environment for reproducing our experiments, 
install the appropriate conda environment that suits your hardware specifications. 
We put forward two YAML environment files: `environment_gpu.yml` CUDA support and `environment.yml` for CPU (and MPS) support.

```commandline
conda env create -f <environment_filename>
```

## Dataset

We provide 8 processed datasets: amazon-beauty, amazon-electro, gowalla, amazon-book, amazon-movies, yelp2018, amazon-cds, citeulike.

## An example to run a 3-layer LightGCN

run LightGCN on **Gowalla** dataset:

* command

` cd code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="gowalla" --topks="[20]" --recdim=64`


*NOTE*:

1. We use wandb to store and plot our results so you might want to create an account for that. More details on how to get started with wandb can be found in their official documentation: https://docs.wandb.ai/quickstart.
2. The code should allow you to reproduce and further extend the experiments we have done. Where you feel necessary add CLI arguments as well.
3. We run everything with seed 2020