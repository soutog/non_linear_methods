import matplotlib.pyplot as plt
import numpy as np


def plot_gd_himmelblau(trajectory, x):
    # Define the range of values for X and Y
    x_range = np.linspace(-6, 6, 100)  # Adjust the range as needed
    y_range = np.linspace(-6, 6, 100)  # Adjust the range as needed

    # Create a grid of points
    X, Y = np.meshgrid(x_range, y_range)

    # Compute Z values using the Himmelblau function
    Z = (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2

    # Plotting the trajectory
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=20, cmap="jet")
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker="o",
        color="white",
        markersize=5,
        label="Trajectory",
    )
    plt.scatter(x[0], x[1], color="red", label="Minimum")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gradient Descent Trajectory - Himmelblau")
    plt.legend()
    plt.colorbar()
    plt.grid(True)
    plt.show()


def plot_gd_rosen(trajectory, x, X, Y, Z):

    # Plotting the trajectory
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=20, cmap="jet")
    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        marker="o",
        color="white",  
        markersize=5,
        label="Trajectory",
    )
    plt.scatter(x[0], x[1], color="red", label="Minimum")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Gradient Descent Trajectory - Himmelblau")
    plt.legend()
    plt.colorbar()
    plt.grid(True)
    plt.show()
