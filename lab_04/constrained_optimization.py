"""
Лабораторная работа №4
Методы условной оптимизации

Реализация:
1. Метод проекции градиента
2. Метод штрафных функций
3. Метод барьерных функций
4. Метод множителей Лагранжа
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import sphere, sphere_gradient, rosenbrock, rosenbrock_gradient
from utils.visualization import plot_convergence, plot_comparison, save_plot


# ========== Функции с ограничениями ==========

def constrained_sphere(x):
    """Сфера с ограничениями."""
    return sphere(x)


def constrained_sphere_gradient(x):
    return sphere_gradient(x)


def constrained_rosenbrock(x):
    """Розенброк с ограничениями."""
    return rosenbrock(x)


def constrained_rosenbrock_gradient(x):
    return rosenbrock_gradient(x)


# ========== Ограничения ==========

def box_constraint(x, lower, upper):
    """
    Проверка ограничения типа коробка.
    Возвращает True если все ограничения выполнены.
    """
    return np.all(x >= lower) and np.all(x <= upper)


def linear_inequality(x, A, b):
    """
    Проверка линейных ограничений-неравенств: Ax <= b
    """
    return np.all(A @ x <= b)


def equality_constraint(x, A, b):
    """
    Проверка ограничений-равенств: Ax = b
    """
    return np.allclose(A @ x, b, atol=1e-6)


# ========== Метод проекции градиента ==========

def project_to_box(x, lower, upper):
    """
    Проекция точки на коробку [lower, upper].
    """
    return np.clip(x, lower, upper)


def project_to_simplex(x):
    """
    Проекция точки на единичный симплекс.
    """
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0)


def projected_gradient_descent(f, grad_f, x0, project_func, 
                                lr=0.01, max_iter=1000, tol=1e-6):
    """
    Метод проекции градиента.
    
    Алгоритм:
    1. Сделать шаг градиентного спуска
    2. Спроецировать результат на допустимое множество
    
    x_{k+1} = P_X(x_k - α * ∇f(x_k))
    
    Параметры:
        project_func: функция проекции на допустимое множество
    """
    x = np.array(x0, dtype=float)
    x = project_func(x)  # Начальная проекция
    history = [f(x)]
    feasible = [True]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # Шаг градиентного спуска
        x_new = x - lr * grad
        
        # Проекция на допустимое множество
        x_new = project_func(x_new)
        
        # Проверка сходимости
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
        history.append(f(x))
        feasible.append(True)
    
    return x, history, feasible


# ========== Метод штрафных функций ==========

def quadratic_penalty(f, grad_f, x0, constraints, mu=1.0, mu_factor=10,
                       max_outer_iter=20, max_inner_iter=100, tol=1e-6):
    """
    Метод квадратичных штрафных функций.
    
    Преобразует задачу с ограничениями в последовательность
    задач без ограничений:
    
    minimize f(x) + μ * sum(max(0, g_i(x))^2)

    Параметры:
        constraints: список функций ограничений g_i(x) <= 0
        mu: начальный параметр штрафа
        mu_factor: коэффициент увеличения штрафа
    """
    x = np.array(x0, dtype=float)
    history = []

    for outer_iter in range(max_outer_iter):
        # Функция с штрафом
        def penalized_f(x):
            penalty = f(x)
            for g in constraints:
                val = g(x)
                if val > 0:  # Нарушение ограничения
                    penalty += mu * val ** 2
            return penalty

        def penalized_grad(x):
            grad = grad_f(x)
            for g in constraints:
                val = g(x)
                if val > 0:
                    # Численный градиент ограничения
                    eps = 1e-7
                    g_grad = np.zeros_like(x)
                    for i in range(len(x)):
                        x_plus = x.copy()
                        x_minus = x.copy()
                        x_plus[i] += eps
                        x_minus[i] -= eps
                        g_grad[i] = (g(x_plus) - g(x_minus)) / (2 * eps)
                    grad += 2 * mu * val * g_grad
            return grad

        # Минимизация с помощью градиентного спуска
        x_inner = x.copy()
        lr = 0.001

        for _ in range(max_inner_iter):
            grad = penalized_grad(x_inner)
            # Обработка NaN/Inf
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                break
            if np.linalg.norm(grad) < tol:
                break
            # Clip градиента для предотвращения переполнения
            grad = np.clip(grad, -1e6, 1e6)
            x_inner = x_inner - lr * grad

        x = x_inner
        history.append(f(x))
        
        # Проверка сходимости
        max_violation = max([max(0, g(x)) for g in constraints], default=0)
        if max_violation < tol:
            break
        
        # Увеличение штрафа
        mu *= mu_factor
    
    return x, history


# ========== Метод барьерных функций ==========

def logarithmic_barrier(f, grad_f, x0, inequality_constraints, 
                         mu=1.0, mu_factor=0.1, max_outer_iter=20,
                         max_inner_iter=100, tol=1e-6):
    """
    Метод логарифмических барьерных функций.
    
    Для ограничений g_i(x) <= 0:
    
    minimize f(x) - μ * sum(log(-g_i(x)))
    
    Параметры:
        inequality_constraints: список функций g_i(x) < 0
        mu: начальный параметр барьера
        mu_factor: коэффициент уменьшения (0 < mu_factor < 1)
    """
    x = np.array(x0, dtype=float)
    history = []
    
    # Проверка начальной точки
    for g in inequality_constraints:
        if g(x) >= 0:
            raise ValueError("Начальная точка должна быть строго допустимой")
    
    for outer_iter in range(max_outer_iter):
        # Функция с барьером
        def barrier_f(x):
            value = f(x)
            for g in inequality_constraints:
                g_val = g(x)
                if g_val >= 0:
                    return float('inf')
                value -= mu * np.log(-g_val)
            return value
        
        def barrier_grad(x):
            grad = grad_f(x)
            for g in inequality_constraints:
                g_val = g(x)
                if g_val >= -1e-10:
                    continue
                # Численный градиент ограничения
                eps = 1e-7
                g_grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    g_grad[i] = (g(x_plus) - g(x_minus)) / (2 * eps)
                grad -= mu * g_grad / g_val
            return grad
        
        # Минимизация
        x_inner = x.copy()
        lr = 0.01
        
        for _ in range(max_inner_iter):
            grad = barrier_grad(x_inner)
            if np.linalg.norm(grad) < tol:
                break
            
            # Backtracking line search
            alpha = 1.0
            while True:
                x_new = x_inner - alpha * grad
                # Проверка допустимости
                feasible = all(g(x_new) < 0 for g in inequality_constraints)
                if feasible and barrier_f(x_new) < barrier_f(x_inner):
                    break
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            
            x_inner = x_new
        
        x = x_inner
        history.append(f(x))
        
        # Уменьшение параметра барьера
        mu *= mu_factor
        
        if mu < 1e-8:
            break
    
    return x, history


# ========== Метод множителей Лагранжа ==========

def augmented_lagrangian(f, grad_f, x0, eq_constraints, ineq_constraints=None,
                          mu=1.0, mu_factor=2.0, max_outer_iter=30,
                          max_inner_iter=200, tol=1e-6):
    """
    Метод расширенного Лагранжиана.

    Для ограничений:
    - равенства: h_j(x) = 0
    - неравенства: g_i(x) <= 0

    L_A(x, λ, μ) = f(x) + Σλ_j*h_j(x) + (μ/2)*Σh_j(x)²
                          + (1/(2μ))*Σ(max(0, λ_i + μ*g_i(x))² - λ_i²)
    """
    x = np.array(x0, dtype=float)
    n_eq = len(eq_constraints) if eq_constraints else 0
    n_ineq = len(ineq_constraints) if ineq_constraints else 0

    # Инициализация множителей Лагранжа
    lambda_eq = np.zeros(n_eq)
    lambda_ineq = np.zeros(n_ineq)

    history = []

    for outer_iter in range(max_outer_iter):
        # Расширенный Лагранжиан
        def aug_lagrangian(x):
            value = f(x)

            # Ограничения-равенства
            for j, h in enumerate(eq_constraints):
                h_val = h(x)
                value += lambda_eq[j] * h_val + (mu / 2) * h_val ** 2

            # Ограничения-неравенства
            if ineq_constraints:
                for i, g in enumerate(ineq_constraints):
                    g_val = g(x)
                    combined = lambda_ineq[i] + mu * g_val
                    if combined > 0:
                        value += (1 / (2 * mu)) * (combined ** 2 - lambda_ineq[i] ** 2)

            return value

        def aug_lagrangian_grad(x):
            grad = grad_f(x)
            
            # Численные градиенты ограничений
            eps = 1e-7
            
            for j, h in enumerate(eq_constraints):
                h_val = h(x)
                h_grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    h_grad[i] = (h(x_plus) - h(x_minus)) / (2 * eps)
                grad += lambda_eq[j] * h_grad + mu * h_val * h_grad
            
            if ineq_constraints:
                for i, g in enumerate(ineq_constraints):
                    g_val = g(x)
                    combined = lambda_ineq[i] + mu * g_val
                    if combined > 0:
                        g_grad = np.zeros_like(x)
                        for k in range(len(x)):
                            x_plus = x.copy()
                            x_minus = x.copy()
                            x_plus[k] += eps
                            x_minus[k] -= eps
                            g_grad[k] = (g(x_plus) - g(x_minus)) / (2 * eps)
                        grad += combined * g_grad
            
            return grad
        
        # Минимизация расширенного Лагранжиана
        x_inner = x.copy()
        lr = 0.01
        
        for _ in range(max_inner_iter):
            grad = aug_lagrangian_grad(x_inner)
            if np.linalg.norm(grad) < tol:
                break
            x_inner = x_inner - lr * grad
        
        x = x_inner
        history.append(f(x))

        # Обновление множителей Лагранжа
        for j, h in enumerate(eq_constraints):
            lambda_eq[j] += mu * h(x)

        if ineq_constraints:
            for i, g in enumerate(ineq_constraints):
                lambda_ineq[i] = max(0, lambda_ineq[i] + mu * g(x))

        # Проверка сходимости
        eq_violation = max([abs(h(x)) for h in eq_constraints], default=0)
        ineq_violation = max([max(0, g(x)) for g in ineq_constraints], default=0) if ineq_constraints else 0

        if eq_violation < tol and ineq_violation < tol:
            break

        # Увеличение штрафа
        mu *= mu_factor

    return x, history, lambda_eq, lambda_ineq


def run_lab4():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №4: Методы условной оптимизации")
    print("=" * 60)
    
    # ========== Пример 1: Оптимизация с ограничениями-коробкой ==========
    print("\n--- Пример 1: Ограничения-коробка ---")
    print("Минимизация сферы с ограничением: -2 <= x_i <= 2")
    
    lower = np.array([-2.0, -2.0])
    upper = np.array([2.0, 2.0])
    x0 = np.array([3.0, 3.0])  # Недопустимая начальная точка
    
    def project_box(x):
        return project_to_box(x, lower, upper)
    
    x_opt, history, feasible = projected_gradient_descent(
        constrained_sphere, constrained_sphere_gradient, x0,
        project_box, lr=0.1, max_iter=100
    )
    
    print(f"Найденное решение: {x_opt}")
    print(f"Значение функции: {constrained_sphere(x_opt):.6f}")
    print(f"В допустимой области: {box_constraint(x_opt, lower, upper)}")
    
    # ========== Пример 2: Метод штрафных функций ==========
    print("\n--- Пример 2: Метод штрафных функций ---")
    print("Минимизация Розенброка с ограничением: x[0] + x[1] <= 1")
    
    x0 = np.array([0.0, 0.0])
    
    # Ограничение g(x) <= 0
    constraint = lambda x: x[0] + x[1] - 1
    
    x_opt, history = quadratic_penalty(
        constrained_rosenbrock, constrained_rosenbrock_gradient, x0,
        constraints=[constraint], mu=1.0, max_outer_iter=15
    )
    
    print(f"Найденное решение: {x_opt}")
    print(f"Значение функции: {constrained_rosenbrock(x_opt):.6f}")
    print(f"Нарушение ограничения: {max(0, constraint(x_opt)):.6f}")
    
    # ========== Пример 3: Метод барьерных функций ==========
    print("\n--- Пример 3: Метод барьерных функций ---")
    print("Минимизация сферы с ограничением: x[0] > 0.5, x[1] > 0.5")
    
    x0 = np.array([1.0, 1.0])  # Строго допустимая точка
    
    # Ограничения g(x) < 0
    ineq_constraints = [
        lambda x: 0.5 - x[0],  # x[0] > 0.5
        lambda x: 0.5 - x[1],  # x[1] > 0.5
    ]
    
    x_opt, history = logarithmic_barrier(
        constrained_sphere, constrained_sphere_gradient, x0,
        inequality_constraints=ineq_constraints, mu=1.0
    )
    
    print(f"Найденное решение: {x_opt}")
    print(f"Значение функции: {constrained_sphere(x_opt):.6f}")

    # ========== Пример 4: Метод множителей Лагранжа ==========
    print("\n--- Пример 4: Метод множителей Лагранжа ---")
    print("Минимизация сферы с ограничением-равенством: x[0] + x[1] = 1")

    x0 = np.array([0.5, 0.5])

    # Ограничение-равенство h(x) = 0
    eq_constraint = lambda x: x[0] + x[1] - 1

    x_opt, history, lambda_eq, lambda_ineq = augmented_lagrangian(
        constrained_sphere, constrained_sphere_gradient, x0,
        eq_constraints=[eq_constraint], ineq_constraints=None,
        mu=1.0, mu_factor=2.0, max_outer_iter=30
    )

    print(f"Найденное решение: {x_opt}")
    print(f"Значение функции: {constrained_sphere(x_opt):.6f}")
    print(f"Нарушение ограничения: {abs(eq_constraint(x_opt)):.6f}")
    print(f"Множитель Лагранжа: {lambda_eq[0]:.6f}")
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    # Сохранение истории для метода штрафных функций
    fig1, _ = plot_convergence(history, 'Метод штрафных функций')
    save_plot(fig1, 'lab4_penalty_method.png', folder='lab_04/output')
    print("Сохранён график: lab_04/output/lab4_penalty_method.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №4 завершена!")
    print("=" * 60)


if __name__ == "__main__":
    run_lab4()
