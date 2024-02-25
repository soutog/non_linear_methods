import time

from functions import himmelblau, rosenbrock
from methods import conjugated_gradient
from plots import plot_method

if __name__ == "__main__":

    print("\n======================================")
    print("CONJUGATED GRADIENT - Himmelblau")

    begin_time = time.time()
    # generate gradient vector
    # Initial guess
    x0 = [-6.0, -6.0]

    # Run gradient descent optimization
    x, iteration, trajectory = conjugated_gradient(himmelblau, x0)
    min_value = himmelblau(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print(f"Initial Point: {x0}")
    print(f"Minimizer:{x}")
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "conjugated_gradient_himmelblau", x, "himmel")

    print("\n======================================")
    print("CONJUGATED GRADIENT - Rosenbrock")

    begin_time = time.time()
    # generate gradient vector
    # Initial guess
    x0 = [2.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = conjugated_gradient(rosenbrock, x0)
    min_value = rosenbrock(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print(f"Initial Point: {x0}")
    print(f"Minimizer:{x}")
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "conjugated_gradient_rosenbrock", x, "rosen")
