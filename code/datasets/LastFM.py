from .BasicDataset import BasicDataset
from os.path import join
from scipy.sparse import csr_matrix

import utils
import pandas as pd
import numpy as np
import torch
import world


class LastFM(BasicDataset):
    def __init__(self, path=join("..", "data", "lastfm")):
        print("loading [last fm]")
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]

        self.n_user = 1892
        self.m_item = 4489

        train_data = pd.read_table(join(path, "data1.txt"), header=None)
        test_data = pd.read_table(join(path, "test1.txt"), header=None)
        trust_net = pd.read_table(
            join(path, "trustnetwork.txt"), header=None).to_numpy()

        trust_net -= 1
        train_data -= 1
        test_data -= 1

        self.trust_net = trust_net
        self.train_data = train_data
        self.test_data = test_data
        self.train_user = np.array(train_data[:][0])
        self.train_unique_users = np.unique(self.train_user)
        self.train_item = np.array(train_data[:][1])
        self.test_user = np.array(test_data[:][0])
        self.test_unique_user = np.unique(self.test_user)
        self.test_item = np.array(test_data[:][1])
        self.graph = None

        dataset_sparsity = len(self.train_user) + len(self.test_user)
        dataset_sparsity /= self.n_user
        dataset_sparsity /= self.m_item

        print(f"LastFm Sparsity : {dataset_sparsity}")

        self.social_net = csr_matrix(
            (np.ones(len(trust_net)), (trust_net[:, 0], trust_net[:, 1])),
            shape=(self.n_users, self.n_users))

        self.user_item_net = csr_matrix(
            (
                np.ones(len(self.train_user)),
                (self.train_user, self.train_item)
            ),
            shape=(self.n_users, self.m_items))

        # pre-calculate
        self.all_positive = self.get_user_pos_items(list(range(self.n_users)))
        self.all_negative = []
        all_items = set(range(self.m_items))

        for i in range(self.n_users):
            pos = set(self.all_positive[i])
            neg = all_items - pos
            self.all_negative.append(np.array(list(neg)))

        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def train_data_size(self):
        return len(self.train_user)

    @property
    def test_dict(self):
        return self.__testDict

    @property
    def all_pos(self):
        return self.all_positive

    def get_sparse_graph(self):
        if self.graph is None:
            user_dim = torch.LongTensor(self.train_user)
            item_dim = torch.LongTensor(self.train_item)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])

            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.graph = torch.sparse.IntTensor(
                index, data,
                torch.Size([self.n_users + self.m_items,
                            self.n_users + self.m_items])
            )

            dense = self.graph.to_dense()

            D = torch.sum(dense, dim=1).float()
            D[D == 0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense / D_sqrt
            dense = dense / D_sqrt.t()

            index = dense.nonzero()
            data = dense[dense >= 1e-9]
            assert len(index) == len(data)

            self.graph = torch.sparse.FloatTensor(
                index.t(), data,
                torch.Size([self.n_users+self.m_items,
                            self.n_users+self.m_items])
            )

            self.graph = self.graph.coalesce().to(world.device)

        return self.graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}

        for i, item in enumerate(self.test_item):
            user = self.test_user[i]

            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]

        return test_data

    def get_user_item_feedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        user_item_feedback = np.array(self.user_item_net[users, items])
        user_item_feedback = user_item_feedback.astype("uint8")
        user_item_feedback = user_item_feedback.reshape((-1, ))

        return user_item_feedback

    def get_user_pos_items(self, users):
        pos_items = []

        for user in users:
            pos_items.append(self.user_item_net[user].nonzero()[1])

        return pos_items

    def __getitem__(self, index):
        user = self.train_unique_users[index]

        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict["test"]

    def __len__(self):
        return len(self.train_unique_users)
