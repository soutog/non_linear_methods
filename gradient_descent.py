import time

from functions import himmelblau, rosenbrock
from methods import gradient_descent
from plots import plot_method

if __name__ == "__main__":

    print("\n======================================")
    print("GRADIENT DESCENT - Himmelblau")

    begin_time = time.time()
    # generate gradient vector
    # Initial guess
    x = [0.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = gradient_descent(himmelblau, x)
    min_value = himmelblau(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "gradient_himmelblau", x, "himmel")

    print("\n======================================")
    print("GRADIENT DESCENT - Rosenbrock")

    begin_time = time.time()
    # generate gradient vector

    # Initial guess
    x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = gradient_descent(rosenbrock, x)
    min_value = rosenbrock(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "gradient_rosenbrock", x, "rosen")
