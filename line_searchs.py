import numdifftools as nd
import numpy as np


def line_search(f, xk, epsilon=1e-6):

    alpha = 1

    while (abs(nd.Gradient(f)(xk)) > epsilon).all():
        desc_direction = -1 * nd.Gradient(f)(xk)
        f_x0 = f(xk + alpha * desc_direction)

        if f_x0 < f(xk):
            xk = xk + alpha * desc_direction
        else:
            alpha *= 0.5

    return alpha


def armijo_search(f, xk, dk, sigma=0.02, gamma=0.5):
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
