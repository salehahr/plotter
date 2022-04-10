import wandb

from tools import data, get_config

config = get_config("configs/edge_nn_baseline.yaml")
api = wandb.Api()

metrics = ["loss", "accuracy", "precision", "recall"]

if __name__ == "__main__":
    if config.multiple_runs:
        runs = config.get_runs(api, all=False)
        for run in runs:
            filename = config.gen_filepath(f"skip_{run.id}")
            data.save_metrics(run, metrics, filename)
    else:
        run = config.get_run(api)
        filename = config.gen_filepath()
        data.save_metrics(run, metrics, filename)
