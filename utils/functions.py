"""
Модуль с тестовыми функциями для оптимизации.
"""

import numpy as np


def sphere(x):
    """
    Функция сфера (quadratic).
    f(x) = sum(x_i^2)
    Глобальный минимум: f(0, 0, ...) = 0
    """
    return np.sum(x ** 2)


def sphere_gradient(x):
    """Градиент функции сфера."""
    return 2 * x


def rosenbrock(x, a=1, b=100):
    """
    Функция Розенброка.
    f(x, y) = (a - x)^2 + b * (y - x^2)^2
    Глобальный минимум: f(1, 1) = 0
    """
    return np.sum((a - x[:-1]) ** 2 + b * (x[1:] - x[:-1] ** 2) ** 2)


def rosenbrock_gradient(x, a=1, b=100):
    """Градиент функции Розенброка."""
    grad = np.zeros_like(x)
    grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    grad[1:-1] = 2 * (x[1:-1] - a) - 4 * b * x[1:-1] * (x[2:] - x[1:-1] ** 2) + \
                 2 * b * (x[1:-1] - x[:-2] ** 2)
    grad[-1] = 2 * b * (x[-1] - x[-2] ** 2)
    return grad


def rastrigin(x, A=10):
    """
    Функция Растригина.
    f(x) = A * n + sum(x_i^2 - A * cos(2 * pi * x_i))
    Глобальный минимум: f(0, 0, ...) = 0
    """
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def rastrigin_gradient(x, A=10):
    """Градиент функции Растригина."""
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


def beale(x):
    """
    Функция Била.
    Глобальный минимум: f(3, 0.5) = 0
    x может быть массивом [x, y] или двумя аргументами
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        x_val, y = x[0], x[1]
    else:
        # Если переданы два аргумента
        x_val = x
        y = x  # Этот случай не используется
    term1 = (1.5 - x_val + x_val * y) ** 2
    term2 = (2.25 - x_val + x_val * y ** 2) ** 2
    term3 = (2.625 - x_val + x_val * y ** 3) ** 2
    return term1 + term2 + term3


def beale_gradient(x):
    """Градиент функции Била."""
    x_val, y = x[0], x[1]
    
    t1 = 1.5 - x_val + x_val * y
    t2 = 2.25 - x_val + x_val * y ** 2
    t3 = 2.625 - x_val + x_val * y ** 3
    
    dx = 2 * t1 * (-1 + y) + 2 * t2 * (-1 + y ** 2) + 2 * t3 * (-1 + y ** 3)
    dy = 2 * t1 * x_val + 2 * t2 * 2 * x_val * y + 2 * t3 * 3 * x_val * y ** 2
    
    return np.array([dx, dy])


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    """
    Функция Акли.
    Глобальный минимум: f(0, 0, ...) = 0
    """
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return term1 + term2 + a + np.e


def ackley_gradient(x, a=20, b=0.2, c=2 * np.pi):
    """Градиент функции Акли."""
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    
    exp1 = np.exp(-b * np.sqrt(sum_sq / n))
    exp2 = np.exp(sum_cos / n)
    
    grad = a * b * exp1 * x / (n * np.sqrt(sum_sq / n)) + \
           exp2 * c * np.sin(c * x) / n
    return grad


def get_function(name):
    """Получить функцию и её градиент по имени."""
    functions = {
        'sphere': (sphere, sphere_gradient),
        'rosenbrock': (rosenbrock, rosenbrock_gradient),
        'rastrigin': (rastrigin, rastrigin_gradient),
        'beale': (lambda x: beale(x[0], x[1]), beale_gradient),
        'ackley': (ackley, ackley_gradient),
    }
    return functions.get(name)
