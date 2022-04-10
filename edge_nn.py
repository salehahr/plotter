import wandb

from tools import get_config, plots

config = get_config("configs/edge_nn.yaml")
api = wandb.Api()

if __name__ == "__main__":
    config.save_project_summary(api)

    tikz_filename = "hyperparam-opt.tex"
    # noinspection PyUnreachableCode
    if __debug__:
        plots.hyperparams(config, show=True, tikz=tikz_filename, backend="pgf")
    else:
        plots.hyperparams(config, show=False, tikz=tikz_filename)
