"""
Лабораторная работа №3
Методы второго порядка

Реализация:
1. Вычисление матрицы Гессе (аналитическое и численное)
2. Классический метод Ньютона
3. Модифицированный метод Ньютона с регуляризацией
4. Квазиньютоновские методы (BFGS, L-BFGS)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import sphere, sphere_gradient, rosenbrock, rosenbrock_gradient, beale, beale_gradient
from utils.visualization import plot_convergence, plot_comparison, save_plot


def numerical_hessian(f, x, epsilon=1e-5):
    """
    Численное вычисление матрицы Гессе методом конечных разностей.
    
    H_ij = (f(x + h*e_i + h*e_j) - f(x + h*e_i - h*e_j) - 
            f(x - h*e_i + h*e_j) + f(x - h*e_i - h*e_j)) / (4*h^2)
    """
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += epsilon
            x_pp[j] += epsilon
            x_pm[i] += epsilon
            x_pm[j] -= epsilon
            x_mp[i] -= epsilon
            x_mp[j] += epsilon
            x_mm[i] -= epsilon
            x_mm[j] -= epsilon
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * epsilon ** 2)
            H[j, i] = H[i, j]
    
    return H


def sphere_hessian(x):
    """Аналитическая матрица Гессе для функции сферы."""
    n = len(x)
    return 2 * np.eye(n)


def rosenbrock_hessian(x):
    """Аналитическая матрица Гессе для функции Розенброка."""
    n = len(x)
    H = np.zeros((n, n))
    
    a, b = 1, 100
    
    H[0, 0] = 2 - 4 * b * (x[1] - x[0] ** 2) + 8 * b * x[0] ** 2
    H[0, 1] = -4 * b * x[0]
    H[1, 0] = H[0, 1]
    
    for i in range(1, n - 1):
        H[i, i] = 2 * (a - x[i]) * 2 * b + 2 * b * (x[i] - x[i-1] ** 2) * 2 + \
                  8 * b * x[i] ** 2 - 4 * b * (x[i+1] - x[i] ** 2)
        H[i, i-1] = -4 * b * x[i-1]
        H[i-1, i] = H[i, i-1]
        H[i, i+1] = -2 * b
        H[i+1, i] = H[i, i+1]
    
    H[-1, -1] = 2 * b
    
    return H


def beale_hessian(x):
    """Аналитическая матрица Гессе для функции Била."""
    x_val, y = x[0], x[1]
    
    t1 = 1.5 - x_val + x_val * y
    t2 = 2.25 - x_val + x_val * y ** 2
    t3 = 2.625 - x_val + x_val * y ** 3
    
    dxx = 2 * (-1 + y) ** 2 + 2 * (-1 + y ** 2) ** 2 + 2 * (-1 + y ** 3) ** 2
    dyy = 2 * x_val ** 2 + 2 * (2 * x_val * y) ** 2 + 4 * t2 * x_val + \
          2 * (3 * x_val * y ** 2) ** 2 + 12 * t3 * x_val * y
    dxy = 2 * (-1 + y) * x_val + 2 * t1 + 2 * (-1 + y ** 2) * 2 * x_val * y + \
          2 * t2 * 2 * y + 2 * (-1 + y ** 3) * 3 * x_val * y ** 2 + 2 * t3 * 3 * y ** 2
    
    return np.array([[dxx, dxy], [dxy, dyy]])


def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """
    Классический метод Ньютона.
    
    x_{k+1} = x_k - H^{-1} * grad_f(x_k)
    
    где H - матрица Гессе.
    """
    x = np.array(x0, dtype=float)
    history = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        H = hess_f(x)
        
        # Решение системы H * delta = -grad
        try:
            delta = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            print("Матрица Гессе вырождена, добавляем регуляризацию")
            H = H + 1e-6 * np.eye(len(x))
            delta = np.linalg.solve(H, -grad)
        
        x = x + delta
        history.append(f(x))
    
    return x, history


def modified_newton(f, grad_f, hess_f, x0, lambda_reg=0.1, max_iter=100, tol=1e-6):
    """
    Модифицированный метод Ньютона с регуляризацией.
    
    x_{k+1} = x_k - (H + λ*I)^{-1} * grad_f(x_k)
    
    Регуляризация обеспечивает положительную определённость матрицы.
    """
    x = np.array(x0, dtype=float)
    history = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        H = hess_f(x)
        
        # Добавляем регуляризацию для положительной определённости
        H_reg = H + lambda_reg * np.eye(len(x))
        
        try:
            delta = np.linalg.solve(H_reg, -grad)
        except np.linalg.LinAlgError:
            # Если всё ещё вырождена, увеличиваем регуляризацию
            H_reg = H + 10 * lambda_reg * np.eye(len(x))
            delta = np.linalg.solve(H_reg, -grad)
        
        x = x + delta
        history.append(f(x))
    
    return x, history


def bfgs(f, grad_f, x0, max_iter=100, tol=1e-6):
    """
    Метод BFGS (Broyden-Fletcher-Goldfarb-Shanno).
    
    Квазиньютоновский метод, аппроксимирующий обратную матрицу Гессе.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)  # Начальная аппроксимация обратной матрицы Гессе
    history = [f(x)]
    grad = grad_f(x)
    
    for i in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        
        # Направление спуска
        p = -H @ grad
        
        # Поиск шага (упрощённый backtracking)
        alpha = 1.0
        c = 0.5
        rho = 0.5
        
        while f(x + alpha * p) > f(x) + c * alpha * grad @ p:
            alpha *= rho
            if alpha < 1e-10:
                break
        
        # Новое положение
        x_new = x + alpha * p
        grad_new = grad_f(x_new)
        
        # Обновление аппроксимации обратной матрицы Гессе
        s = x_new - x
        y = grad_new - grad
        
        if np.dot(s, y) > 1e-10:  # Проверка на положительную определённость
            rho_bfgs = 1.0 / np.dot(s, y)
            I = np.eye(n)
            V = I - rho_bfgs * np.outer(s, y)
            H = V @ H @ V.T + rho_bfgs * np.outer(s, s)
        
        x = x_new
        grad = grad_new
        history.append(f(x))
    
    return x, history


def lbfgs(f, grad_f, x0, m=10, max_iter=100, tol=1e-6):
    """
    L-BFGS (Limited-memory BFGS).
    
    Версия BFGS для задач большой размерности.
    Хранит только последние m пар (s, y).
    """
    x = np.array(x0, dtype=float)
    history = [f(x)]
    grad = grad_f(x)
    
    s_history = []  # История разностей x
    y_history = []  # История разностей градиентов
    rho_history = []  # История 1 / (y^T s)
    
    for i in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        
        # Вычисление направления с помощью two-loop recursion
        q = grad.copy()
        alpha_list = []
        
        # Первый проход (backwards)
        for j in range(len(s_history) - 1, -1, -1):
            alpha = rho_history[j] * np.dot(s_history[j], q)
            alpha_list.insert(0, alpha)
            q = q - alpha * y_history[j]
        
        # Начальное приближение обратной матрицы Гессе
        if len(s_history) > 0:
            gamma = np.dot(s_history[-1], y_history[-1]) / np.dot(y_history[-1], y_history[-1])
        else:
            gamma = 1.0
        
        r = gamma * q
        
        # Второй проход (forwards)
        for j in range(len(s_history)):
            beta = rho_history[j] * np.dot(y_history[j], r)
            r = r + (alpha_list[j] - beta) * s_history[j]
        
        p = -r
        
        # Поиск шага
        alpha = 1.0
        c = 0.5
        rho = 0.5
        
        while f(x + alpha * p) > f(x) + c * alpha * grad @ p:
            alpha *= rho
            if alpha < 1e-10:
                break
        
        # Обновление
        x_new = x + alpha * p
        grad_new = grad_f(x_new)
        
        s = x_new - x
        y = grad_new - grad
        
        if np.dot(s, y) > 1e-10:
            # Удаляем старые записи если нужно
            if len(s_history) >= m:
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
            
            s_history.append(s)
            y_history.append(y)
            rho_history.append(1.0 / np.dot(s, y))
        
        x = x_new
        grad = grad_new
        history.append(f(x))
    
    return x, history


def run_lab3():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №3: Методы второго порядка")
    print("=" * 60)
    
    # Параметры для тестирования
    x0_sphere = np.array([5.0, 5.0])
    x0_rosen = np.array([2.0, 2.0])
    x0_beale = np.array([3.5, 0.6])
    
    # Методы для сравнения
    methods_sphere = {
        'Newton': lambda f, g, h, x0: newton_method(f, g, h, x0, max_iter=50),
        'Modified Newton': lambda f, g, h, x0: modified_newton(f, g, h, x0, max_iter=50),
        'BFGS': lambda f, g, h, x0: bfgs(f, g, x0, max_iter=50),
        'L-BFGS': lambda f, g, h, x0: lbfgs(f, g, x0, max_iter=50),
    }
    
    all_results = {}
    
    # ========== Тест 1: Функция сферы ==========
    print("\n--- Тест 1: Функция сферы ---")
    histories_sphere = []
    labels = list(methods_sphere.keys())
    
    for name, method in methods_sphere.items():
        x_opt, history = method(sphere, sphere_gradient, sphere_hessian, x0_sphere)
        print(f"{name:18s}: f(x) = {sphere(x_opt):.2e}, итераций: {len(history)}")
        histories_sphere.append(history)
    
    all_results['sphere'] = (histories_sphere, labels)
    
    # ========== Тест 2: Функция Розенброка ==========
    print("\n--- Тест 2: Функция Розенброка ---")
    histories_rosen = []
    
    for name, method in methods_sphere.items():
        x_opt, history = method(rosenbrock, rosenbrock_gradient, rosenbrock_hessian, x0_rosen)
        print(f"{name:18s}: f(x) = {rosenbrock(x_opt):.2e}, итераций: {len(history)}")
        histories_rosen.append(history)
    
    all_results['rosenbrock'] = (histories_rosen, labels)
    
    # ========== Тест 3: Функция Била ==========
    print("\n--- Тест 3: Функция Била ---")
    histories_beale = []
    
    for name, method in methods_sphere.items():
        x_opt, history = method(beale, beale_gradient, beale_hessian, x0_beale)
        print(f"{name:18s}: f(x) = {beale(x_opt):.2e}, итераций: {len(history)}")
        histories_beale.append(history)
    
    all_results['beale'] = (histories_beale, labels)
    
    # ========== Сравнение с методами первого порядка ==========
    print("\n--- Сравнение с градиентным спуском ---")
    
    from lab_01.gradient_descent import gradient_descent

    gd_histories_sphere = []
    gd_labels = []

    for lr in [0.01, 0.1, 0.5]:
        _, history, _ = gradient_descent(sphere, sphere_gradient, x0_sphere,
                                          step_size=lr, max_iter=50)
        gd_histories_sphere.append(history)
        gd_labels.append(f'GD (lr={lr})')
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    # Сравнение методов второго порядка
    fig1, _ = plot_comparison(
        histories_sphere, labels,
        'Методы второго порядка - Функция сферы'
    )
    save_plot(fig1, 'lab3_sphere_second_order.png', folder='lab_03/output')
    print("Сохранён график: lab_03/output/lab3_sphere_second_order.png")
    
    fig2, _ = plot_comparison(
        histories_rosen, labels,
        'Методы второго порядка - Функция Розенброка'
    )
    save_plot(fig2, 'lab3_rosenbrock_second_order.png', folder='lab_03/output')
    print("Сохранён график: lab_03/output/lab3_rosenbrock_second_order.png")
    
    fig3, _ = plot_comparison(
        histories_beale, labels,
        'Методы второго порядка - Функция Била'
    )
    save_plot(fig3, 'lab3_beale_second_order.png', folder='lab_03/output')
    print("Сохранён график: lab_03/output/lab3_beale_second_order.png")
    
    # Сравнение с градиентным спуском
    fig4, _ = plot_comparison(
        histories_sphere + gd_histories_sphere,
        labels + gd_labels,
        'Сравнение методов 1-го и 2-го порядка - Сфера'
    )
    save_plot(fig4, 'lab3_comparison.png', folder='lab_03/output')
    print("Сохранён график: lab_03/output/lab3_comparison.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №3 завершена!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_lab3()
