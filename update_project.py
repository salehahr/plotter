import wandb

from tools import Config

config = Config("configs/edge_nn.yaml")
api = wandb.Api()


def update_config(config_: Config):
    runs = ["gxmn7ntn", "skaoai90"]

    update_vals = {
        "adjacency_frac": 0.2,
        "train_imgs": 1000,
        # "batch_norm": True
    }
    config_.update_config(api, update_vals, runs)


if __name__ == "__main__":
    config.update_metric(api, "val_precision")
