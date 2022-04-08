from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import pandas as pd

from tools import filter, plots

if TYPE_CHECKING:
    from tools.config import Config


def get_losses(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["loss", "val_loss"]
    return metrics[columns]


def save_metrics(
    data: pd.DataFrame,
    config: Config,
    filename: Optional[str] = None,
    smooth: bool = True,
) -> None:
    filepath = _generate_filepath(config, filename)

    # smooth data
    if smooth:
        smoothed = data.apply(
            lambda s: filter.gaussian_smooth(s, points=10, stdev=1)
        ).add_suffix("_sm")

        # noinspection PyUnreachableCode
        if __debug__:
            plots.plot_smoothed(data["val_loss"], smoothed["val_loss_sm"])

        data = pd.concat([data, smoothed], axis=1)

    # noinspection PyTypeChecker
    data.to_csv(filepath, index_label="epoch", float_format="%.4f")


def _generate_filepath(config: Config, filename: Optional[str] = None) -> str:
    # parse filename
    filename = (
        f"{config.run_name}_{filename}" if filename is not None else config.run_name
    )
    if not filename.endswith("csv"):
        filename += ".csv"

    return os.path.join(config.folder_path, filename)
