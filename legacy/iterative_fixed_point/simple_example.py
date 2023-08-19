import numpy as np

def f1(x):
    return x * x - 0.35 * x * x * x - np.sin(x) + x

# x = 0.85
# x = 2.285

for iteration_id in range(1_000):
    x = f1(x)
    print(iteration_id, x)
print(x)
