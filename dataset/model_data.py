from dataclasses import dataclass

import numpy as np


@dataclass
class ModelData:
    state: np.ndarray
    open_price: np.ndarray
    close_price: np.ndarray
    symbols_cnt: int
    days_cnt: int