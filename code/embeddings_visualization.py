import torch
from datasets import Loader
import os
import world
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if not os.path.exists("emb_visuals"):
        os.makedirs("emb_visuals", exist_ok=True)

    for emb_file in os.listdir(world.EMBS_PATH):
        _, layer, method, dataset = emb_file.split("_")[: 4]

        world.dataset = dataset
        dataloader = Loader(
            config=world.config, path=os.path.join("..", "data", dataset)
        )
        graph = dataloader.get_sparse_graph()
        num_users = dataloader.n_user
        emb_file_path = os.path.join(world.EMBS_PATH, emb_file)
        embeddings = torch.load(emb_file_path).cpu().numpy()

        embeddings = TSNE(n_components=2).fit_transform(embeddings)

        user_embeddings = embeddings[:num_users]
        item_embeddings = embeddings[num_users:]

        plt.scatter(
            user_embeddings[:, 0],
            user_embeddings[:, 1],
            c="r",
            s=10,
            marker="o"
        )
        plt.scatter(
            item_embeddings[:, 0],
            item_embeddings[:, 1],
            c="b",
            s=10,
            marker="o"
        )

        plt.legend(["Users", "Items"])

        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig(
            os.path.join("emb_visuals", f"{layer}_{method}_{dataset}.png"))

        plt.clf()
