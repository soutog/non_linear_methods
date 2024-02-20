import time
import numdifftools as nd
import numpy as np
from plots import plot_gd_himmelblau, plot_gd_rosen


def armijo(func, grad_func, x, dx, alpha=1.0, beta=0.5, c=0.1):
    """
    Armijo line search algorithm for finding a step size satisfying
    sufficient decrease condition.

    Parameters:
        func (callable): Objective function.
        grad_func (callable): Gradient of the objective function.
        x (array_like): Current point.
        dx (array_like): Search direction.
        alpha (float): Initial step size.
        beta (float): Fraction by which step size is reduced.
        c (float): Constant for sufficient decrease condition.

    Returns:
        float: Step size satisfying Armijo condition.
    """
    while func(x + alpha * dx) > func(x) + c * alpha * grad_func(x).dot(dx):
        alpha *= beta
    return alpha


def gradient_descent(f, df, x, learning_rate=0.01, epsilon=1e-6, max_iter=100000):
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

        learning_rate = armijo(f, gradient, x, learning_rate)
        x_new = x - learning_rate * gradient

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

    # print("\n======================================")
    # print("GRADIENT DESCENT - Himmelblau")

    # begin_time = time.time()
    # # generate gradient vector
    # himmelblau_gradient = nd.Gradient(himmelblau)

    # # Initial guess
    # x = [0.0, 0.0]

    # # Run gradient descent optimization
    # x, iteration, trajectory = gradient_descent(himmelblau, himmelblau_gradient, x)
    # min_value = himmelblau(x)
    # print(himmelblau_gradient([x]))

    # end_time = time.time()

    # print(f"{iteration} iterations")
    # print("Minimizer:", x)
    # print("Minimum value:", min_value)
    # print(f"Total Time Gradient Descent: {end_time-begin_time}")
    # print(f"Average Iteration Time Gradient Descent: {(end_time-begin_time)/iteration}")

    # # plot_gd_himmelblau(trajectory, x)

    print("\n======================================")
    print("GRADIENT DESCENT - Rosenbrock")

    begin_time = time.time()
    # generate gradient vector
    rosenbrock_gradient = nd.Gradient(rosenbrock)

    # Initial guess
    x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = gradient_descent(rosenbrock, rosenbrock_gradient, x, learning_rate=0.001)
    min_value = rosenbrock(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time Gradient Descent: {end_time-begin_time}")
    print(f"Average Iteration Time Gradient Descent: {(end_time-begin_time)/iteration}")

    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    Z = rosenbrock([X, Y])
    plot_gd_rosen(trajectory, x, X, Y, Z)
