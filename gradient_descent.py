import time
import numdifftools as nd
import numpy as np
from plots import plot_gd_himmelblau, plot_gd_rosen

 
def line_search(f, xk, epsilon=1e-6):

    alpha = 1

    while (abs(nd.Gradient(f)(xk)) > epsilon).all():
        desc_direction = -1*nd.Gradient(f)(xk)
        f_x0 = f(xk + alpha * desc_direction)

        if f_x0 < f(xk):
            xk = xk + alpha * desc_direction
        else:
            alpha *= 0.5

    return alpha


def Armijo_Search(f, xk, dk, sigma=0.02, gamma=0.5):
    """
    Busca linear com Condição de Armijo (1° Condição de Wolfe) - Desigualdade 1, que consiste em fazer um decréscimo da f proporcional ao tamanho do passo

    Parâmetros
    ----------------------------------------
    f : callable
        Função objetivo (função custo).
    xk : array
        Ponto atual.
    dk : array
        Direção de descida (grad_T(f(x_{k})) . d_{k}  < 0).
    sigma : float, opcional
        Valor de sigma entre (0, 1) - constante de decréscimo. Padrão = 0.02
    gamma : float, opcional
        Valor de gamma entre (0, 1). Padrão = 0.5

    Saída
    ----------------------------------------
    lambda : float
        Valor de lambda que satisfaz a condição de Armijo.
    f_x0 : float
        Valor de f no ponto x_{k+1}.
    """

    # Começa constante lambda = 1
    lambda_ = 1

    # Direção de descida d = grad_{T}(f(x_{k})) . dk < 0
    desc_direction = np.dot(nd.Gradient(f)(xk), dk)
    f_x0 = f(xk + lambda_ * dk)

    # Calcula a função em um ponto menor que x_k com passo inicial lambda = 1
    # Condição de Armijo para determinar o tamanho do passo, diminuindo a função custo
    while f_x0 > f(xk) + sigma * lambda_ * desc_direction:

        lambda_ = gamma * lambda_
        f_x0 = f(xk + lambda_ * dk)

    # Retorna o passo lambda
    return lambda_


def gradient_descent(f, df, x, sigma=0.02, epsilon=1e-6, max_iter=100000):
    """
    Gradient Descent Optimization for a Univariate Function

    Parameters:
        f (function): The objective function to minimize.
        df (function): The derivative of the objective function.
        x0 (float): Initial guess for the minimum.
        sigma (float): Constante de descréscimo de Armijo
        epsilon (float): The convergence criterion.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: The minimum value found and the corresponding x value.
    """

    count = 0
    trajectory = [x]

    for i in range(max_iter):
        gradient = -1*df([x])
        
        # Linear Search with Armijo Condition
        learning_rate = Armijo_Search(f, x, gradient, sigma)
        # learning_rate = line_search(f, x)
        # learning_rate = 0.01
        x_new = x + learning_rate * gradient
        if (abs(x_new - x) < 1e-6).all():
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
    x, iteration, trajectory = gradient_descent(himmelblau, himmelblau_gradient, x)
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
    x, iteration, trajectory = gradient_descent(rosenbrock, rosenbrock_gradient, x)
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
    # plot_gd_rosen(trajectory, x, X, Y, Z)
