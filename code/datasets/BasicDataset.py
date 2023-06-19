from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def test_dict(self):
        raise NotImplementedError

    @property
    def all_pos(self):
        raise NotImplementedError

    def get_user_item_feedback(self, users, items):
        raise NotImplementedError

    def get_user_pos_items(self, users):
        raise NotImplementedError

    def get_sparse_graph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
