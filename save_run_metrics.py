import wandb

from tools import data, get_config

config = get_config("configs/nodes_nn.yaml")
api = wandb.Api()

# metrics = ["loss", "accuracy", "precision", "recall"]
metrics = [
    "loss",
    "node_pos_loss",
    "degrees_loss",
    "node_types_loss",
    "node_pos_accuracy",
    "degrees_accuracy",
    "node_types_accuracy",
]

if __name__ == "__main__":
    if config.multiple_runs:
        runs = config.get_runs(api, all=False)
        for run in runs:
            filename = config.gen_filepath(f"{run.id}")
            data.save_metrics(run, metrics, filename)
    else:
        run = config.get_run(api)
        filename = config.gen_filepath()
        data.save_metrics(run, metrics, filename)
