import os
import world
import dataloader
import model
from pprint import pprint
from world import DATA_PATH

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'amazon-beauty', 'amazon-movies', 'amazon-cds', 'amazon-electro', 'movielens', 'citeulike']:
    dataset = dataloader.Loader(path=os.path.join(DATA_PATH, world.dataset))
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
    # dataset = dataloader.Loader(path=os.path.join(DATA_PATH, world.dataset))


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}