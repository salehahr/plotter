import wandb

from tools import data, get_config

config = get_config("configs/edge_nn_baseline.yaml")
api = wandb.Api()

if __name__ == "__main__":
    metrics = config.get_metrics(api)
    losses = data.get_losses(metrics)
    data.save_metrics(losses, config)
