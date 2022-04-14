from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pandas as pd

from tools import filter, plots

if TYPE_CHECKING:
    import wandb

    from tools.data_types import Number, ParamsDict


def get_losses(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["loss", "val_loss"]
    return metrics[columns]


def get_runs_df(runs: wandb.apis.public.Runs) -> pd.DataFrame:
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

    return {_parse_key(k): v for k, v in summary.items()}


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
    training_params = {
        _parse_key(k): run.config[k] for k in training_keys if k in params.keys()
    }
    model_arch_params = {
        _parse_key(k): run.config[k] for k in model_arch_keys if k in params.keys()
    }

    return training_params, model_arch_params


def _parse_key(key: str) -> str:
    """For LaTeX. Removes underscores, inserts predefined LaTeX macros."""
    macros = {
        "batch_norm": r"\bn{}",
        "n_filters": r"$\filtsym$",
        "n_conv2_blocks": r"$\nctwo$",
        "n_conv3_blocks": r"$\ncthr$",
    }

    shortcuts = {"best_val_": "best val. ", "_": " "}

    for k in macros.keys():
        if key == k:
            return macros[k]

    for s in shortcuts:
        if s in key:
            key = key.replace(s, shortcuts[s])

    return key


def parse_booleans(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.applymap(type) != bool
    d = {True: r"\true", False: r"\false"}
    return df.where(mask, df.replace(d))


def flatten(df: pd.DataFrame, cols_to_flatten: List[str]) -> pd.DataFrame:
    df = pd.concat(
        [
            df.drop(cols_to_flatten, axis=1),
            *[df[col].apply(pd.Series) for col in cols_to_flatten],
        ],
        axis=1,
    )

    return parse_booleans(df)


def save_metrics(
    run: wandb.apis.public.Run,
    metrics: List[str],
    filepath: Optional[str] = None,
    smooth: bool = True,
) -> None:
    metrics = metrics + [f"val_{m}" for m in metrics]

    history = run.history()
    data = history[metrics]

    # smooth data
    if smooth:
        smoothed = data.apply(
            lambda s: filter.gaussian_smooth(s, points=10, stdev=1.5)
        ).add_suffix("_sm")

        # noinspection PyUnreachableCode
        if __debug__:
            plots.plot_smoothed(data["val_loss"], smoothed["val_loss_sm"])

        data = pd.concat([data, smoothed], axis=1)

    # noinspection PyTypeChecker
    data.to_csv(filepath, index_label="epoch", float_format="%.4f")


def save_flat(df: pd.DataFrame, filepath: str, **kwargs) -> None:
    cols_to_flatten = ["model_arch_params", "train_params", "summary"]
    flat_df = flatten(df, cols_to_flatten)

    # noinspection PyTypeChecker
    flat_df.to_csv(filepath, **kwargs)
