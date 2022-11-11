import numpy as np

def dstack_product(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
