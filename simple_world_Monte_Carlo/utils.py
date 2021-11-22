from typing import List

import numpy as np


def dstack_product(x: List, y: List) -> np.ndarray:
    """
    Takes to list and returns their Cartesian product.
    """
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
