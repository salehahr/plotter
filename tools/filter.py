from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import filters
from scipy.signal import gaussian

if TYPE_CHECKING:
    from tools.data_types import TimeSeries


def smooth(vals: list, k: int = 10) -> np.ndarray:
    arr_plc = np.zeros((len(vals),))
    arr = np.array(vals)

    for i, a in enumerate(arr):
        if i < k:
            arr_plc[i] = np.mean(arr[0:k])
        else:
            arr_plc[i] = np.mean(arr[i - k + 1 : i + 1])

    return arr_plc


def rolling(s: TimeSeries) -> TimeSeries:
    return s.rolling(window=10).mean().fillna(method="bfill")


def gaussian_smooth(s: TimeSeries, points: int = 10, stdev: int = 1) -> TimeSeries:
    b = gaussian(points, stdev)
    return filters.convolve1d(s, b / b.sum())
