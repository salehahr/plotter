from typing import TYPE_CHECKING, Dict, Union

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    TimeSeries = Union[np.ndarray, pd.Series]

    Number = Union[int, float]
    ParamVal = Union[str, Number, bool]
    ParamsDict = Dict[str, ParamVal]
