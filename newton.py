import time

from functions import himmelblau, rosenbrock
from methods import newton
from plots import plot_method

if __name__ == "__main__":

    print("\n======================================")
    print("NEWTON - Himmelblau")

    begin_time = time.time()
    # generate gradient vector

    # Initial guess
    x0 = [2.0, 1.0]

    # Run gradient descent optimization
    x, iteration, trajectory = newton(himmelblau, x0)
    min_value = himmelblau(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print(f"Initial point: {x0}")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "newton_himmelblau", x, "himmel")

    print("\n======================================")
    print("NEWTON - Rosenbrock")

    begin_time = time.time()
    # generate gradient vector

    # Initial guess
    x0 = [2.0, 1.0]

    # Run gradient descent optimization
    x, iteration, trajectory = newton(rosenbrock, x0)
    min_value = rosenbrock(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print(f"Initial point: {x0}")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "newton_rosenbrock", x, "rosen")
