from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib.path import Path

if TYPE_CHECKING:
    from tools.config import Config
    from tools.data_types import TimeSeries

COLWIDTH = 442.11  # pt


def set_size(
    width_pt: float, fraction: int = 1, subplots: Tuple[int, int] = (1, 1)
) -> Tuple[float, float]:
    """Set figure dimensions to sit nicely in our document.
    Source: https://jwalton.info/Matplotlib-latex-PGF/

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in


def plot_smoothed(orig: TimeSeries, smoothed: TimeSeries) -> None:
    plt.plot(orig)
    plt.plot(smoothed)
    plt.show()


def hyperparams(
    config: Config,
    show: bool = False,
    tikz: Optional[str] = None,
    backend: Optional[str] = None,
) -> None:
    """
    Modified from source: https://stackoverflow.com/questions/8230638/.
    """
    if backend == "pgf":
        mpl.use("pgf")
        mpl.rcParams.update(
            {
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )

    # get only mature runs, convert booleans
    df = pd.read_csv(config.sweeps_csv)
    df = df[df["epoch"] == 53]
    df[r"\bn{}"] = df[r"\bn{}"].map({r"\true": 1, r"\false": 0})

    # sort so that higher precision values are the last/in front when plotting
    df = df.sort_values(by="best val. precision", ascending=True)

    # column settings
    metric_key = "best val. precision"
    min_val, max_val = 0.7, 1.0
    columns = [r"\bn{}", r"$\nctwo$", r"$\ncthr$", r"$\filtsym$", metric_key]
    column_labels = ["bn", "$nc_2$", "$nc_3$", "$f$", metric_key]
    column_ticks = [
        [0, 1],
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 4, 5, 6],
        np.linspace(min_val, max_val, 4 + 1),
    ]

    num_params = len(columns)
    num_runs = len(df)

    # column fit parameters
    col_mins, col_maxs, col_diffs = _get_col_params(df[columns])

    # transform data to size of leftmost column
    Z = _transform_tuning_data(df[columns], col_mins, col_diffs)

    figsize = set_size(COLWIDTH)
    fig, (host, col_ax) = plt.subplots(
        1, 2, gridspec_kw={"width_ratios": [30, 1]}, figsize=figsize
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
    host.set_xticklabels(column_labels)
    host.tick_params(axis="x", which="major", direction="in", pad=7)
    host.xaxis.tick_top()

    colours, colourbar = _set_colours(col_ax, min_val, max_val)
    for r in range(num_runs):
        best_metric = df[metric_key].iloc[r]
        colour = colours(best_metric)

        # straight lines
        # host.plot(range(num_params), Z[r, :], c=colour)

        # beziers
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
        patch = patches.PathPatch(path, facecolor="none", lw=1.5, edgecolor=colour)

        host.add_patch(patch)

    host.set_title("Model Architecture Hyperparameters", pad=20)
    plt.tight_layout()

    if tikz:
        filepath = _get_save_filepath(tikz)
        if backend == "pgf":
            plt.savefig(filepath.replace("tex", "pgf"))

            # noinspection PyUnreachableCode
            # compile and preview latex file
            if __debug__:
                compile_tikz("main-pgf.tex")
            return
        elif backend == "tpl":
            save_tikz(filepath, fig)
        else:
            raise Exception

    if show:
        plt.show()


def _get_save_filepath(filename: str) -> str:
    # noinspection PyUnreachableCode
    if __debug__:
        dir_name = "test"
    else:
        dir_name = "results"

    return os.path.join(os.getcwd(), dir_name, filename)


def save_tikz(filepath: str, figure):
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
        compile_tikz("main.tex")


def compile_tikz(filename: str) -> None:
    dir_name = "test"
    filepath = os.path.join(os.getcwd(), dir_name, filename)
    os.system(
        "pdflatex -interaction=nonstopmode -synctex=1 -output-format=pdf "
        + f"-output-directory={dir_name} {filepath}"
    )
    os.system(filepath.replace(".tex", ".pdf"))


def _set_colours(
    ax, min_val: float, max_val: float
) -> Tuple[Callable, plt.cm.ScalarMappable]:
    cm_divisions = 100
    cmap = plt.get_cmap("plasma_r", cm_divisions)

    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    colourbar = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="vertical"
    )

    def colours_normed(metric: float) -> np.ndarray:
        x = int(norm(metric) * 100)
        return cmap.colors[x]

    return colours_normed, colourbar


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
