import time

from functions import himmelblau, rosenbrock
from methods import modified_newton
from plots import plot_method

if __name__ == "__main__":

    print("\n======================================")
    print("MODIFIED NEWTON - Himmelblau")

    begin_time = time.time()
    # generate gradient vector
    # Initial guess
    x0 = [-6.0, -6.0]

    # Run gradient descent optimization
    x, iteration, trajectory = modified_newton(himmelblau, x0)
    min_value = himmelblau(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print(f"Initial Point: {x0}")
    print(f"Minimizer:{x}")
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "modified_newton_himmelblau", x, "himmel")

    print("\n======================================")
    print("MODIFIED NEWTON - Rosenbrock")

    begin_time = time.time()
    # generate gradient vector
    # Initial guess
    x0 = [0.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = modified_newton(rosenbrock, x0)
    min_value = rosenbrock(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print(f"Initial Point: {x0}")
    print(f"Minimizer:{x}")
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "modified_newton_rosenbrock", x, "rosen")
