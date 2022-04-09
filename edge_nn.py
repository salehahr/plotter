import wandb

from tools import get_config

config = get_config("configs/edge_nn.yaml")
api = wandb.Api()

if __name__ == "__main__":
    config.save_project_summary(api)
