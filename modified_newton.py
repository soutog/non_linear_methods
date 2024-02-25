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
    x = [0.0, 0.0]

    # Run gradient descent optimization
    x, iteration, trajectory = modified_newton(himmelblau, x)
    min_value = himmelblau(x)

    end_time = time.time()

    print(f"{iteration} iterations")
    print("Minimizer:", x)
    print("Minimum value:", min_value)
    print(f"Total Time: {end_time-begin_time}")
    print(f"Average Iteration Time: {(end_time-begin_time)/iteration}")

    plot_method(trajectory, "gradient_himmelblau", x, "himmel")
