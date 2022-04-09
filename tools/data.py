from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from tools import filter, plots

if TYPE_CHECKING:
    import wandb

    from tools.config import Config
    from tools.data_types import Number, ParamsDict


def get_losses(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["loss", "val_loss"]
    return metrics[columns]


def get_runs_df(runs: wandb.apis.public.Runs):
    ids, names, train_params, model_arch_params, summary = [], [], [], [], []
    for run in runs:
        ids.append(run.id)
        names.append(run.name)

        params = get_run_params(run)
        train_params.append(params[0])
        model_arch_params.append(params[1])

        summary.append(get_run_summary(run))

    return pd.DataFrame(
        {
            "id": ids,
            "name": names,
            "summary": summary,
            "train_params": train_params,
            "model_arch_params": model_arch_params,
        }
    )


# noinspection PyProtectedMember
def get_run_summary(run: wandb.apis.public.Run) -> Dict[str, Number]:
    summary = run.summary._json_dict

    # remove metrics
    metric_keys = [i for k in summary.keys() if k.startswith("val") for i in (k, k[4:])]
    for mk in metric_keys:
        del summary[mk]

    # remove unwanted keys
    for k in ["graph", "_wandb"]:
        summary.pop(k, None)

    return summary


def get_run_params(run: wandb.apis.public.Run) -> Tuple[ParamsDict, ParamsDict]:
    training_keys = [
        "optimiser",
        "learning_rate",
        "batch_size",
        "node_pairs_in_batch",
        "adjacency_frac",
        "train_imgs",
        "epoch",
    ]
    model_arch_keys = [
        "batch_norm",
        "n_filters",
        "n_conv2_blocks",
        "n_conv3_blocks",
        "depth",
    ]

    params = run.config
    training_params = {k: run.config[k] for k in training_keys if k in params.keys()}
    model_arch_params = {
        k: run.config[k] for k in model_arch_keys if k in params.keys()
    }

    return training_params, model_arch_params


def flatten(df: pd.DataFrame, cols_to_flatten: List[str]) -> pd.DataFrame:
    return pd.concat(
        [
            df.drop(cols_to_flatten, axis=1),
            *[df[col].apply(pd.Series) for col in cols_to_flatten],
        ],
        axis=1,
    )


def save_project_summary(df: pd.DataFrame, config: Config) -> None:
    # general
    save(df, config, "runs.csv")

    # sweep
    sweep_mask = df["name"].str.contains("sweep")
    sweeps_df = df[sweep_mask].drop("train_params", axis=1)
    sweeps_filepath = config.gen_filepath("sweeps.csv")
    save_flat(sweeps_df, config, sweeps_filepath)


def save_metrics(
    data: pd.DataFrame,
    config: Config,
    filename: Optional[str] = None,
    smooth: bool = True,
) -> None:
    # smooth data
    if smooth:
        smoothed = data.apply(
            lambda s: filter.gaussian_smooth(s, points=10, stdev=1)
        ).add_suffix("_sm")

        # noinspection PyUnreachableCode
        if __debug__:
            plots.plot_smoothed(data["val_loss"], smoothed["val_loss_sm"])

        data = pd.concat([data, smoothed], axis=1)

    save(data, config, filename, index_label="epoch", float_format="%.4f")


def save_flat(df: pd.DataFrame, config: Config, filename: str, **kwargs) -> None:
    filepath = config.gen_filepath(filename)

    cols_to_flatten = ["model_arch_params", "summary"]
    flat_df = flatten(df, cols_to_flatten)

    # noinspection PyTypeChecker
    flat_df.to_csv(filepath, **kwargs)


def save(data: pd.DataFrame, config: Config, filename: str, **kwargs) -> None:
    filepath = config.gen_filepath(filename)
    # noinspection PyTypeChecker
    data.to_csv(filepath, **kwargs)
