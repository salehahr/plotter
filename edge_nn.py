import wandb

from tools import get_config, plots

config = get_config("configs/edge_nn.yaml")
api = wandb.Api()

if __name__ == "__main__":
    config.save_project_summary(api)

    df = config.get_sweeps_df(api, config.sweep_id)

    tikz_filename = "hyperparam-opt.tex"
    # noinspection PyUnreachableCode
    if __debug__:
        plots.hyperparams(df, show=True, tikz=tikz_filename, backend='pgf')
    else:
        plots.hyperparams(df, show=False, tikz=tikz_filename)
