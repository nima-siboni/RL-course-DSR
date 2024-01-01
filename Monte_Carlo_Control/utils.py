"""Some general utils"""
from typing import List

import numpy as np


def return_all_the_state(x_values: List, y_values: List) -> np.ndarray:
    """
    Takes two lists and returns their Cartesian product.
    """
    return np.dstack(np.meshgrid(x_values, y_values)).reshape(-1, 2)
