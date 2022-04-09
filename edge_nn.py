import wandb

from tools import get_config, plots

config = get_config("configs/edge_nn.yaml")
api = wandb.Api()

if __name__ == "__main__":
    config.save_project_summary(api)

    df = config.get_sweeps_df(api, config.sweep_id)
    plots.hyperparams(df)
