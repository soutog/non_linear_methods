import matplotlib.pyplot as plt
import numpy as np

from functions import himmelblau, rosenbrock


def plot_method(trajectory, plot_name, x, func):

    if func == "himmel":
        # Define the range of values for X and Y
        x_range = np.linspace(-6, 6, 100)  # Adjust the range as needed
        y_range = np.linspace(-6, 6, 100)  # Adjust the range as needed

        # Create a grid of points
        X, Y = np.meshgrid(x_range, y_range)

        # Compute Z values using the Himmelblau function
        # Z = (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2
        Z = himmelblau([X, Y])

        # Plotting the trajectory
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=20, cmap="jet")
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="white",
            markersize=5,
            label="Trajectory",
        )
        plt.scatter(x[0], x[1], color="red", label="Minimum")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Himmelblau")
        plt.legend()
        plt.colorbar()
        plt.grid(True)
        plt.savefig(f"{plot_name}.png")
        # plt.show()

    elif func == "rosen":
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-6, 6, 100)
        X, Y = np.meshgrid(x, y)

        Z = rosenbrock([X, Y])

        # Plotting the trajectory
        plt.figure(figsize=(8, 6))
        plt.contourf(X, Y, Z, levels=20, cmap="jet")
        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color="white",
            markersize=5,
            label="Trajectory",
        )
        plt.scatter(x[0], x[1], color="red", label="Minimum")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Rosenbrock")
        plt.legend()
        plt.colorbar()
        plt.grid(True)
        plt.savefig(f"{plot_name}.png")
        # plt.show()
