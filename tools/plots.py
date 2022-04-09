from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib.path import Path

if TYPE_CHECKING:
    from tools.data_types import TimeSeries


def plot_smoothed(orig: TimeSeries, smoothed: TimeSeries) -> None:
    plt.plot(orig)
    plt.plot(smoothed)
    plt.show()


def hyperparams(
    df: pd.DataFrame, show: bool = False, tikz: Optional[str] = None
) -> None:
    """
    Modified from source: https://stackoverflow.com/questions/8230638/.
    """
    columns = ["n_conv2_blocks", "n_conv3_blocks", "n_filters", "best_val_precision"]
    column_ticks = [[1, 2, 3], [1, 2, 3], [2, 3, 4, 5, 6], np.linspace(0, 1, 5 + 1)]

    num_params = len(columns)
    num_runs = len(df)

    # column fit parameters
    col_mins, col_maxs, col_diffs = _get_col_params(df[columns])

    # transform data to size of leftmost column
    Z = _transform_tuning_data(df[columns], col_mins, col_diffs)

    fig, (host, col_ax) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [30, 1]}, figsize=[10, 5]
    )
    axes = [host] + [host.twinx() for _ in range(num_params - 1)]

    # vertical axis settings
    host.spines["right"].set_visible(False)
    for i, ax in enumerate(axes):
        ax.set_ylim(col_mins[i], col_maxs[i])

        # remove top and bottom border
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # parameter ticks
        ax.set_yticks(column_ticks[i])

        if ax != host:
            ax.spines["left"].set_visible(False)

            ax.spines["right"].set_visible(True)
            ax.spines["right"].set_position(("axes", i / (num_params - 1)))
            ax.yaxis.set_ticks_position("right")
            plt.setp(ax.get_yticklabels(), backgroundcolor="white")

    # horizontal axis settings
    host.set_xlim(0, num_params - 1)
    host.set_xticks(range(num_params))
    host.set_xticklabels(columns)
    host.tick_params(axis="x", which="major", direction="in", pad=7)
    host.xaxis.tick_top()

    colours, colourbar = _set_colours(col_ax)
    for r in range(num_runs):
        best_val_prec = df["best_val_precision"].iloc[r]
        colour = colours[int(best_val_prec * 100)]

        # straight line
        # host.plot(range(num_params), Z[r, :], c=colour)

        # bezier
        verts = list(
            zip(
                [
                    x
                    for x in np.linspace(
                        0, num_runs - 1, num_runs * 3 - 2, endpoint=True
                    )
                ],
                np.repeat(Z[r, :], 3)[1:-1],
            )
        )
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]

        # noinspection PyTypeChecker
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor="none", lw=1, edgecolor=colour)

        host.add_patch(patch)

    host.set_title("Model Architecture Hyperparameters", pad=20)
    plt.tight_layout()

    if tikz:
        save_tikz(tikz, fig)

    if show:
        plt.show()


def save_tikz(filename: str, figure):
    # noinspection PyUnreachableCode
    if __debug__:
        dir_name = "test"
    else:
        dir_name = "results"

    filepath = os.path.join(os.getcwd(), dir_name, filename)
    tikzplotlib.clean_figure()
    tikzplotlib.save(
        filepath,
        figure=figure,
        axis_width=r"\w",
        axis_height=r"\h",
        strict=True,
        wrap=False,
    )

    # noinspection PyUnreachableCode
    # compile and preview latex file
    if __debug__:
        main_fp = os.path.join(os.getcwd(), dir_name, "main.tex")
        os.system(
            "pdflatex -interaction=nonstopmode -synctex=1 -output-format=pdf "
            + f"-output-directory={dir_name} {main_fp}"
        )
        os.system(main_fp.replace(".tex", ".pdf"))


def _set_colours(ax) -> Tuple[np.ndarray, plt.cm.ScalarMappable]:
    cm_divisions = 100
    cmap = plt.get_cmap("plasma_r", cm_divisions)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    colourbar = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="vertical"
    )

    return cmap.colors, colourbar


def _transform_tuning_data(
    df: pd.Dataframe, col_mins: np.ndarray, col_diffs: np.ndarray
) -> np.ndarray:
    X = df.to_numpy()
    Z = np.zeros_like(X)
    Z[:, 0] = X[:, 0]
    Z[:, 1:] = (X[:, 1:] - col_mins[1:]) / col_diffs[1:] * col_diffs[0] + col_mins[0]

    return Z


def _get_col_params(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    # get range of columns
    col_mins = df.min().to_numpy()
    col_maxs = df.max().to_numpy()
    col_diffs = col_maxs - col_mins

    # pad the columns
    col_mins -= col_diffs * 0.05
    col_maxs += col_diffs * 0.05
    col_diffs = col_maxs - col_mins

    return col_mins, col_maxs, col_diffs
