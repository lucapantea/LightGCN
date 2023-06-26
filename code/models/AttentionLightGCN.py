import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LightGCN:
    """
    Light Graph Convolutional Network (LightGCN) model.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (BasicDataset): The dataset containing user-item interactions.

    Attributes:
        config (dict): Configuration parameters for the model.
        dataset (BasicDataset): The dataset containing user-item interactions.
        embs (torch.Tensor or None): Embedding matrix of shape
                                   (num_users + num_items, latent_dim) or None.
        num_users (int): Number of unique users in the dataset.
        num_items (int): Number of unique items in the dataset.
        latent_dim (int): Dimensionality of the latent space.
        n_layers (int): Number of layers in LightGCN.
        keep_prob (float): Dropout keep probability.
        a_split (bool): Whether to split the adjacency matrix or not.
        embedding_user (torch.nn.Embedding): User embedding layer.
        embedding_item (torch.nn.Embedding): Item embedding layer.
        sigmoid (torch.nn.Sigmoid): Sigmoid activation function.
        graph (torch.sparse.FloatTensor): Sparse graph representation.
    """

    def __init__(self, config, dataset):
        """
        Initialize the LightGCN model.

        Args:
            config (dict): Configuration parameters for the model.
            dataset (BasicDataset): The dataset containing user-item
                                    interactions.
        """
        self.config = config
        self.dataset = dataset
        self.embs = None
        self.__init_weight()

    def get_embedding_matrix(self):
        """
        Get the embedding matrix.

        Returns:
            torch.Tensor or None: The embedding matrix of shape
            (num_users + num_items, latent_dim) or None if not available.
        """
        return self.embs

    def __init_weight(self):
        """
        Initialize the user and item embedding weights based on the
        configuration parameters.
        """
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config["latent_dim_rec"]
        self.n_layers = self.config["lightGCN_n_layers"]
        self.keep_prob = self.config["keep_prob"]
        self.a_split = self.config["A_split"]
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        if self.config["pretrain"] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print("Using NORMAL distribution initializer.")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config["user_emb"]))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config["item_emb"]))
            print("Using pretrained data.")

        self.sigmoid = nn.Sigmoid()
        self.graph = self.dataset.get_sparse_graph()
        self.embs = None

        print(f"LightGCN is ready to go (dropout: {self.config['dropout']}).")

    @staticmethod
    def __dropout_x(x, keep_prob):
        """
        Apply dropout to the sparse tensor x with keep probability keep_prob.

        Args:
            x (torch.sparse.FloatTensor): Sparse tensor to apply dropout on.
            keep_prob (float): Dropout keep probability.

        Returns:
            torch.sparse.FloatTensor: The dropout-applied sparse tensor.
        """
        size = x.size()
        index = x.indices().t()
        values = x.values()

        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()

        index = index[random_index]
        values = values[random_index] / keep_prob

        g = torch.sparse.FloatTensor(index.t(), values, size)

        return g

    def __dropout(self, keep_prob):
        """
        Apply dropout to the graph representation with keep probability
        keep_prob.

        Args:
            keep_prob (float): Dropout keep probability.

        Returns:
            list or torch.sparse.FloatTensor: The dropout-applied graph 
                                              representation.
        """
        if self.a_split:
            graph = []
            for g in self.graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.graph, keep_prob)

        return graph

    def forward(self):
        """
        Forward pass of the LightGCN model.

        Returns:
            tuple: A tuple of all users' embeddings and all items' embeddings.
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config["dropout"]:
            if self.training:
                print("Dropping.")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph

        for _ in range(self.n_layers):
            if self.a_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))

                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        if self.config["save_embs"]:
            self.embs = embs

        if self.config["single"]:
            light_out = embs[:, -1, :].squeeze()
        else:
            light_out = torch.mean(embs, dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings

    def get_user_rating(self, users):
        """
        Get the predicted ratings for a given list of users.

        Args:
            users (torch.Tensor): Tensor containing user indices.

        Returns:
            torch.Tensor: Predicted ratings for the users, with values 
                          between 0 and 1.
        """
        all_users, all_items = self.forward()

        users_emb = all_users[users.long()]
        items_emb = all_items

        rating = self.sigmoid(torch.matmul(users_emb, items_emb.t()))

        return rating


class BaseAttention(LightGCN):
    """
    Light Graph Convolutional Network (LightGCN) model with a simple attention 
    mechanism.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (BasicDataset): The dataset containing user-item interactions.

    Attributes:
        attention_weights (torch.nn.Parameter): Attention weights parameter.
    """

    def __init__(self, config, dataset):
        """
        Initialize the BaseAttention model.

        Args:
            config (dict): Configuration parameters for the model.
            dataset (BasicDataset): The dataset containing user-item 
                                    interactions.
        """
        super().__init__(config, dataset)
        self.attention_weights = torch.nn.Parameter(
            torch.randn(config["lightGCN_n_layers"] + 1),
            requires_grad=True
        )

    def forward(self):
        """
        Forward pass of the BaseAttention model.

        Returns:
            tuple: A tuple of all users' embeddings and all items' embeddings.
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config["dropout"]:
            if self.training:
                print("Dropping.")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph

        for _ in range(self.n_layers):
            if self.a_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))

                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        if self.config["save_embs"]:
            self.embs = embs

        # Attention Mechanism
        attention_scores = F.softmax(self.attention_weights, dim=0)
        light_out = torch.sum(embs * attention_scores.view(1, -1, 1), dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings

    def parameters_norm(self):
        """
        Compute the norm of the model"s parameters.

        Returns:
            float: The norm of the model"s parameters.
        """
        return super().parameters_norm() + \
            self.attention_weights.norm(2).pow(2)


class FinerAttention(LightGCN):
    """
    Light Graph Convolutional Network (LightGCN) model with a finer attention
    mechanism.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (BasicDataset): The dataset containing user-item interactions.

    Attributes:
        attention_weights_users (torch.nn.Parameter): Attention weights
                                                      parameter for users.
        attention_weights_items (torch.nn.Parameter): Attention weights
                                                      parameter for items.
    """

    def __init__(self, config, dataset):
        """
        Initialize the FinerAttention model.

        Args:
            config (dict): Configuration parameters for the model.
            dataset (BasicDataset): The dataset containing user-item
                                    interactions.
        """
        super().__init__(config, dataset)
        self.attention_weights_users = torch.nn.Parameter(
            torch.randn(config["lightGCN_n_layers"] + 1),
            requires_grad=True)
        self.attention_weights_items = torch.nn.Parameter(
            torch.randn(config["lightGCN_n_layers"] + 1),
            requires_grad=True)

    def forward(self):
        """
        Forward pass of the FinerAttention model.

        Returns:
            tuple: A tuple of all users" embeddings and all items" embeddings.
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config["dropout"]:
            if self.training:
                print("Dropping.")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph

        for _ in range(self.n_layers):
            if self.a_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))

                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        if self.config["save_embs"]:
            self.embs = embs

        # Calculate the softmax
        attention_scores_users = F.softmax(
            self.attention_weights_users, dim=0).view(1, -1, 1)
        attention_scores_items = F.softmax(
            self.attention_weights_items, dim=0).view(1, -1, 1)

        # Create block diagonal attention matrix
        attention_scores = torch.cat([
            attention_scores_users.repeat(self.num_users, 1, 1),
            attention_scores_items.repeat(self.num_items, 1, 1)], dim=0)

        # Compute weighted sum in one step
        light_out = torch.sum(embs * attention_scores, dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings

    def parameters_norm(self):
        """
        Compute the norm of the model"s parameters.

        Returns:
            float: The norm of the model"s parameters.
        """
        return super().parameters_norm() + \
            self.attention_weights_users.norm(2).pow(2) + \
            self.attention_weights_items.norm(2).pow(2)


class ScaledDotProductAttentionLightGCN(LightGCN):
    """
    Light Graph Convolutional Network (LightGCN) model with scaled dot-product 
    attention.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (BasicDataset): The dataset containing user-item interactions.
    """

    def forward(self):
        """
        Forward pass of the ScaledDotProductAttentionLightGCN model.

        Returns:
            tuple: A tuple of all users" embeddings and all items" embeddings.
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config["dropout"]:
            if self.training:
                print("Dropping.")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.graph
        else:
            g_droped = self.graph

        for _ in range(self.n_layers):
            if self.a_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))

                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        if self.config["save_embs"]:
            self.embs = embs

        queries, keys, values = self.prepare_attention_inputs(embs)
        attention_output = self.compute_attention(queries, keys, values)

        if self.config["single"]:
            light_out = attention_output[:, -1, :].squeeze()
        else:
            light_out = torch.mean(attention_output, dim=1)

        all_users_embeddings, all_items_embeddings = torch.split(
            light_out, [self.num_users, self.num_items])

        return all_users_embeddings, all_items_embeddings

    def prepare_attention_inputs(self, embs):
        """
        Prepare inputs for the attention mechanism.

        Args:
            embs (torch.Tensor): Embeddings tensor.

        Returns:
            tuple: A tuple of queries, keys, and values.
        """
        return embs, embs, embs

    @staticmethod
    def compute_attention(queries, keys, values):
        """
        Compute the scaled dot-product attention.

        Args:
            queries (torch.Tensor): Queries tensor.
            keys (torch.Tensor): Keys tensor.
            values (torch.Tensor): Values tensor.

        Returns:
            torch.Tensor: The attention output tensor.
        """
        scaling_factor = math.sqrt(queries.size(-1))
        attention_scores = torch.matmul(
            queries, keys.transpose(-2, -1)) / scaling_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)
        return attention_output


class WeightedScaledDotProductAttentionLightGCN(ScaledDotProductAttentionLightGCN):
    """
    ScaledDotProductAttentionLightGCN model with linear projections to
    embeddings.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (BasicDataset): The dataset containing user-item interactions.
    """

    def __init__(self, config, dataset):
        """
        Initialize the WeightedScaledDotProductAttentionLightGCN model.

        Args:
            config (dict): Configuration parameters for the model.
            dataset (BasicDataset): The dataset containing user-item
                                    interactions.
        """
        super().__init__(config, dataset)
        self.latent_dim = self.config["latent_dim_rec"]
        self.attention_dim = self.config.get("attention_dim", 1)
        self.query_projection = nn.Linear(
            self.latent_dim, self.attention_dim, bias=False)
        self.key_projection = nn.Linear(
            self.latent_dim, self.attention_dim, bias=False)
        self.value_projection = nn.Linear(
            self.latent_dim, self.attention_dim, bias=False)

    def prepare_attention_inputs(self, embs):
        """
        Prepare inputs for the attention mechanism.

        Args:
            embs (torch.Tensor): Embeddings tensor.

        Returns:
            tuple: A tuple of queries, keys, and values.
        """
        queries = self.query_projection(embs)
        keys = self.key_projection(embs)
        values = self.value_projection(embs)
        return queries, keys, values

    def parameters_norm(self):
        """
        Compute the norm of the model"s parameters.

        Returns:
            float: The norm of the model"s parameters.
        """
        return super().parameters_norm() + \
            self.query_projection.weight.norm(2) + \
            self.key_projection.weight.norm(2) + \
            self.value_projection.weight.norm(2)
