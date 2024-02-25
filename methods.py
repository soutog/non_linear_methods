import numdifftools as nd
import numpy as np
from line_searchs import armijo_search, line_search
import copy


def gradient_descent(f, x, sigma=0.02, epsilon=1e-6, max_iter=100000):
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

    iter = 0
    trajectory = [x]
    df = nd.Gradient(f)

    for i in range(max_iter):

        gradient = -1 * df(x)

        # Linear Search with Armijo Condition
        learning_rate = armijo_search(f, x, gradient, sigma)
        # learning_rate = line_search(f, x)
        x_new = x + learning_rate * gradient
        if (abs(x_new - x) < epsilon).all():
            break

        iter += 1
        x = x_new
        trajectory.append(x)

    return x, iter, np.array(trajectory)


def newton(f, x, sigma=0.02, epsilon=1e-6, max_iter=100000):

    df = nd.Gradient(f)
    f_hessian = nd.Hessian(f)
    grad_fk = df(x)
    grad_fk_norm = np.linalg.norm(df(x))
    f_hessian_xk = f_hessian(x)

    trajectory = [x]
    iter = 0

    for _ in range(max_iter):
        gradient = -1 * df(x)

        if (
            (np.linalg.norm(f(x)) > 1e16)
            or (grad_fk_norm > 1e16)
            or (np.linalg.norm(f_hessian_xk) > 1e16)
        ):
            print("\nErro: Overflow\n")
            break

        # Não invertível
        if abs(np.linalg.det(f_hessian_xk)) <= 1e-3:
            break

        pk = -np.linalg.solve(f_hessian(x), df(x))
        learning_rate = 1

        # Se estiver longe do ponto, a matriz Hessiana pode ser não definida positiva.
        # Se isso acontecer, seguir pelo Método do Gradiente:
        if not (
            np.dot(pk, grad_fk) < (-0.001 * np.linalg.norm(df(x)) * np.linalg.norm(pk))
        ):
            # gradient
            gradient = -1 * df(x)

            # Linear Search with Armijo Condition
            learning_rate = armijo_search(f, x, gradient, sigma)
        else:
            # Linear Search with Armijo Condition
            learning_rate = armijo_search(f, x, pk, sigma)

        x_new = x + learning_rate * pk
        if (abs(x_new - x) < epsilon).all():
            break

        iter += 1
        x = x_new
        trajectory.append(x)

    if abs(np.linalg.det(f_hessian_xk)) <= 1e-3:
        print("\nMatriz Hessiana não invertível. Mudar o ponto inicial.\n")
        return [], [], []

    elif np.min(np.linalg.eigvals(f_hessian_xk)) <= 1e-3:
        print(
            "\nMatriz Hessiana não PD. Mudar o ponto inicial. Ponto de sela encontrado.\n"
        )

    return x, iter, np.array(trajectory)


def modified_newton(f, x, sigma=0.02, epsilon=1e-6, max_iter=100000):

    df = nd.Gradient(f)
    f_hessian = nd.Hessian(f)

    trajectory = [x]
    iter = 0

    for _ in range(max_iter):
        # gradient = -1 * df(x)

        identity_matrix = np.eye(len(x)) * 0.001
        f_hessian_modified = f_hessian(x) + identity_matrix

        pk = -np.linalg.solve(f_hessian_modified, df(x))

        learning_rate = armijo_search(f, x, pk, sigma)
        # learning_rate = line_search(f, x)

        x_new = x + learning_rate * pk
        if (abs(x_new - x) < epsilon).all():
            break

        iter += 1
        x = x_new
        trajectory.append(x)

    return x, iter, np.array(trajectory)


def conjugated_gradient(f, x, sigma=0.02, epsilon=1e-6, max_iter=100000):
    iter = 0
    trajectory = [x]
    df = nd.Gradient(f)
    x0 = copy.deepcopy(x)

    for i in range(max_iter):

        if i == 0:
            gradient = -1 * df(x0)

            learning_rate = armijo_search(f, x0, gradient, sigma)
            # learning_rate = line_search(f, x)

            x_new = x0 + learning_rate * gradient

            if (abs(x_new - x0) < epsilon).all():
                break

            iter += 1
            x = x_new
            trajectory.append(x)
        else:
            x1 = copy.deepcopy(x0)  # anterior
            xk = copy.deepcopy(x)  # corrente

            gradient = -1 * df(xk)

            pk = gradient + (
                np.dot(np.transpose(df(xk) - df(x1)), df(xk))
                / np.dot(np.transpose(df(x1)), df(x1))
            ) * df(x1)

            learning_rate = armijo_search(f, x, pk, sigma)
            # learning_rate = line_search(f, x)

            x_new = x + learning_rate * pk

            if (abs(x_new - x) < epsilon).all():
                break

            iter += 1
            x1 = copy.deepcopy(xk)
            x = copy.deepcopy(x_new)
            trajectory.append(x)

    return x, iter, np.array(trajectory)
