import numpy as np


def rosenbrock(x, a=1, b=100):
    """
    Rosenbrock function

    Parameters:
        x (array-like): Input vector of length N.
        a (float): Parameter (default 1).
        b (float): Parameter (default 100).

    Returns:
        float: Value of the Rosenbrock function at x.
    """
    x = np.asarray(x)  # Convert input to NumPy array if it's not already
    N = len(x)
    sum_term = 0
    for i in range(N - 1):
        sum_term += (a - x[i]) ** 2 + b * (x[i + 1] - x[i] ** 2) ** 2
    return sum_term


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
