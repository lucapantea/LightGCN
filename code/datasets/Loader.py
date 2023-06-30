from .BasicDataset import BasicDataset
import world
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import torch
from time import time
import scipy.sparse as sp
from os.path import join


class Loader(BasicDataset):
    def __init__(self, config, path):
        super().__init__()
        print(f"loading [{path}]")

        self.split = config["A_split"]
        self.folds = config["adj_matrix_folds"]
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.n_user = 0
        self.m_item = 0
        train_file = join(path, "train.txt")
        test_file = join(path, "test.txt")
        self.path = path
        train_unique_users, train_item, train_user = [], [], []
        test_unique_users, test_item, test_user = [], [], []
        self.train_data_size = 0
        self.test_data_size = 0
        self.config = config

        # Maintains a mapping based on interactions {user_id -> [item_ids]}
        self.user_interactions_dict_train = defaultdict(list)

        with open(train_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")

                    items = [int(i) for i in line[1:]]
                    uid = int(line[0])

                    train_unique_users.append(uid)
                    train_user.extend([uid] * len(items))
                    train_item.extend(items)

                    # Maintain a mapping of user_id -> [item_ids]
                    self.user_interactions_dict_train[uid].extend(items)

                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.train_data_size += len(items)

        self.train_unique_users = np.array(train_unique_users)
        self.train_user = np.array(train_user)
        self.train_item = np.array(train_item)

        with open(test_file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip("\n").split(" ")

                    try:
                        items = [int(i) for i in line[1:]]
                    except Exception:
                        continue

                    uid = int(line[0])

                    test_unique_users.append(uid)
                    test_user.extend([uid] * len(items))
                    test_item.extend(items)
                    
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)

                    self.test_data_size += len(items)

        self.m_item += 1
        self.n_user += 1

        self.test_unique_users = np.array(test_unique_users)

        self.test_user = np.array(test_user)
        self.test_item = np.array(test_item)

        self.Graph = None
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.test_data_size} interactions for testing")

        dataset_sparsity = self.train_data_size + self.test_data_size
        dataset_sparsity /= self.n_users
        dataset_sparsity /= self.m_items
        print(f"{world.dataset} Sparsity : {dataset_sparsity}")

        self.user_item_net = csr_matrix(
            (
                np.ones(len(self.train_user)),
                (self.train_user, self.train_item)
            ),
            shape=(self.n_user, self.m_item)
        )

        self.users_D = np.array(self.user_item_net.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1

        self.items_D = np.array(self.user_item_net.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        self._allPos = self.get_user_pos_items(list(range(self.n_user)))
        self.__testDict = self.__build_test()

        # Assing users to bins based on the number of interactions they have
        self.user_bins_by_num_interactions = self.distribute_users_into_bins_by_num_interactions(num_bins=world.num_bins)

        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def test_dict(self):
        return self.__testDict

    @property
    def all_pos(self):
        return self._allPos

    def __split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds

        for i_fold in range(self.folds):
            start = i_fold*fold_len

            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold.append(
                self.__convert_sp_mat_to_sp_tensor(
                    A[start: end]).coalesce().to(world.device)
            )

        return A_fold

    def __convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)

        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()

        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)

        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            # try:
            #     pre_adj_mat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
            #     print("successfully loaded...")
            #     norm_adj = pre_adj_mat
            # except Exception:
            print("generating adjacency matrix")
            # start_time = time()

            num_nodes = self.n_users + self.m_items
            adj_mat = sp.dok_matrix(
                (num_nodes, num_nodes), dtype=np.float32)
            adj_mat = adj_mat.tolil()

            R = self.user_item_net.tolil()
            adj_mat[: self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, : self.n_users] = R.T
            adj_mat = adj_mat.todok()

            if not self.config["l1"] and (self.config["side_norm"].lower() == "l" or self.config["side_norm"].lower() == "r"):
                rowsum = np.array(np.square(adj_mat).sum(axis=1))
                # rowsum = np.square(np.array(adj_mat.sum(axis=1))
                # rowsum_squared = np.square(adj_mat).sum(axis=1)
                # rowsum = np.sqrt(rowsum_squared)
            else:
                rowsum = np.array(adj_mat.sum(axis=1))

            # L1 normalization
            exponent = -1 if self.config["l1"] else -0.5
            d_inv = np.power(rowsum, exponent).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            # left normalization
            if self.config["side_norm"].lower() == "l":
                norm_adj = d_mat.dot(adj_mat)

            # right normalization
            elif self.config["side_norm"].lower() == "r":
                norm_adj = adj_mat.dot(d_mat)

            # symmetric normalization
            elif self.config["side_norm"].lower() == "both":
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)

            norm_adj = norm_adj.tocsr()

            # end_time = time()
            # print(f"costing {end_time-start_time}s, saved norm_mat...")
            # sp.save_npz(join(self.path, "s_pre_adj_mat.npz"), norm_adj)

            if self.split:
                self.Graph = self.__split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self.__convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")

        return self.Graph

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
        user_item_feedback = user_item_feedback.reshape((-1,))

        return user_item_feedback

    def get_user_pos_items(self, users):
        posItems = []

        for user in users:
            posItems.append(self.user_item_net[user].nonzero()[1])

        return posItems
    
    def distribute_users_into_bins_by_num_interactions(self, num_bins):
        log_values = [np.log(len(self.user_interactions_dict_train[user])) for user in self.user_interactions_dict_train.keys()]

        # Create bins
        min_num_interactions = min(log_values)
        max_num_interactions = max(log_values)
        bin_thresholds = np.linspace(min_num_interactions, max_num_interactions, num_bins)

        # Assign users to a bin based on the number of items they interacted with
        bin_indices = np.digitize(log_values, bin_thresholds, right=True)

        # Create a dictionary that maps users to bins
        user_bin_dict = dict(zip(self.user_interactions_dict_train.keys(), bin_indices))
        return user_bin_dict
