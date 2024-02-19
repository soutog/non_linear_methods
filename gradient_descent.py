import time
import numdifftools as nd
import numpy as np
from plots import plot_gd_himmelblau


def gradient_descent(df, x, learning_rate=0.01, epsilon=1e-6, max_iter=1000):
    """
    Gradient Descent Optimization for a Univariate Function

    Parameters:
        f (function): The objective function to minimize.
        df (function): The derivative of the objective function.
        x0 (float): Initial guess for the minimum.
        learning_rate (float): The step size for each iteration.
        epsilon (float): The convergence criterion.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: The minimum value found and the corresponding x value.
    """

    count = 0
    trajectory = [x]

    for i in range(max_iter):
        gradient = df([x])

        x_new = x + learning_rate * gradient

        if (abs(x_new - x) < 1e-06).all():
            break

        count += 1
        x = x_new
        trajectory.append(x)

    return x, count, np.array(trajectory)


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


if __name__ == "__main__":

    print("\n======================================")
    print("GRADIENT DESCENT - Himmelblau")

    begin_time = time.time()
    # generate gradient vector
    himmelblau_gradient = nd.Gradient(himmelblau)

    # Initial guess
    x = [0.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = gradient_descent(himmelblau_gradient, x)
    min_value = himmelblau(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time Gradient Descent: {end_time-begin_time}")
    print(f"Average Iteration Time Gradient Descent: {(end_time-begin_time)/iteration}")

    # plot_gd_himmelblau(trajectory, x)

    print("\n======================================")
    print("GRADIENT DESCENT - Rosenbrock")

    begin_time = time.time()
    # generate gradient vector
    rosenbrock_gradient = nd.Gradient(rosenbrock)

    # Initial guess
    x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = gradient_descent(rosenbrock_gradient, x)
    min_value = rosenbrock_gradient(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time Gradient Descent: {end_time-begin_time}")
    print(f"Average Iteration Time Gradient Descent: {(end_time-begin_time)/iteration}")

    # plot_gd_himmelblau(trajectory, x)
