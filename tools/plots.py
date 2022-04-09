from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from tools.data_types import TimeSeries


def plot_smoothed(orig: TimeSeries, smoothed: TimeSeries):
    plt.plot(orig)
    plt.plot(smoothed)
    plt.show()
