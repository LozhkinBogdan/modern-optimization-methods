"""
Лабораторная работа №1
Градиентные методы первого порядка

Реализация:
1. Метод градиентного спуска с постоянным шагом
2. Метод градиентного спуска с адаптивным шагом (backtracking line search)
"""

import numpy as np
import sys
import os

# Добавляем корневую директорию в путь для импорта utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import sphere, sphere_gradient, rosenbrock, rosenbrock_gradient
from utils.visualization import plot_convergence, plot_trajectory, plot_comparison, save_plot


def gradient_descent(f, grad_f, x0, step_size=0.01, max_iter=1000, tol=1e-6):
    """
    Градиентный спуск с постоянным шагом.
    
    Параметры:
        f: целевая функция
        grad_f: градиент целевой функции
        x0: начальная точка
        step_size: размер шага (learning rate)
        max_iter: максимальное количество итераций
        tol: точность по норме градиента
    
    Возвращает:
        x_opt: найденная точка минимума
        history: история значений функции
        trajectory: траектория движения
    """
    x = np.array(x0, dtype=float)
    history = [f(x)]
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        # Проверка на сходимость
        if np.linalg.norm(grad) < tol:
            print(f"Сходимость достигнута на итерации {i}")
            break
        
        # Шаг градиентного спуска
        x = x - step_size * grad
        
        history.append(f(x))
        trajectory.append(x.copy())
    
    return x, history, trajectory


def gradient_descent_adaptive(f, grad_f, x0, alpha=0.5, beta=0.5, 
                               max_iter=1000, tol=1e-6, max_step=1.0):
    """
    Градиентный спуск с адаптивным выбором шага (backtracking line search).
    
    Алгоритм:
    1. Начинаем с шага max_step
    2. Уменьшаем шаг умножением на beta, пока не выполнится условие Армихо:
       f(x - t*grad) <= f(x) - alpha * t * ||grad||^2
    
    Параметры:
        f: целевая функция
        grad_f: градиент целевой функции
        x0: начальная точка
        alpha: параметр условия Армихо (0 < alpha < 0.5)
        beta: параметр уменьшения шага (0 < beta < 1)
        max_iter: максимальное количество итераций
        tol: точность по норме градиента
        max_step: начальный размер шага
    
    Возвращает:
        x_opt: найденная точка минимума
        history: история значений функции
        trajectory: траектория движения
    """
    x = np.array(x0, dtype=float)
    history = [f(x)]
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        grad_norm_sq = np.dot(grad, grad)
        
        # Проверка на сходимость
        if np.sqrt(grad_norm_sq) < tol:
            print(f"Сходимость достигнута на итерации {i}")
            break
        
        # Backtracking line search
        step = max_step
        f_x = f(x)
        
        while f(x - step * grad) > f_x - alpha * step * grad_norm_sq:
            step *= beta
            if step < 1e-10:  # Защита от слишком маленького шага
                break
        
        # Шаг градиентного спуска
        x = x - step * grad
        
        history.append(f(x))
        trajectory.append(x.copy())
    
    return x, history, trajectory


def run_lab1():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №1: Градиентные методы первого порядка")
    print("=" * 60)
    
    # Параметры для тестирования
    x0_sphere = np.array([5.0, 5.0])
    x0_rosen = np.array([2.0, 2.0])
    
    results = []
    
    # ========== Тест 1: Функция сферы ==========
    print("\n--- Тест 1: Функция сферы ---")
    print("Начальная точка:", x0_sphere)
    
    # Градиентный спуск с постоянным шагом
    print("\n1. Градиентный спуск с постоянным шагом (lr=0.1)")
    x_opt1, hist1, traj1 = gradient_descent(
        sphere, sphere_gradient, x0_sphere, step_size=0.1, max_iter=100
    )
    print(f"Найденный минимум: {x_opt1}")
    print(f"Значение функции: {sphere(x_opt1):.10f}")
    print(f"Итераций: {len(hist1)}")
    results.append(('Sphere, constant step', hist1))
    
    # Градиентный спуск с адаптивным шагом
    print("\n2. Градиентный спуск с адаптивным шагом")
    x_opt2, hist2, traj2 = gradient_descent_adaptive(
        sphere, sphere_gradient, x0_sphere, max_iter=100
    )
    print(f"Найденный минимум: {x_opt2}")
    print(f"Значение функции: {sphere(x_opt2):.10f}")
    print(f"Итераций: {len(hist2)}")
    results.append(('Sphere, adaptive step', hist2))
    
    # ========== Тест 2: Функция Розенброка ==========
    print("\n--- Тест 2: Функция Розенброка ---")
    print("Начальная точка:", x0_rosen)
    
    # Градиентный спуск с постоянным шагом
    print("\n1. Градиентный спуск с постоянным шагом (lr=0.001)")
    x_opt3, hist3, traj3 = gradient_descent(
        rosenbrock, rosenbrock_gradient, x0_rosen, step_size=0.001, max_iter=1000
    )
    print(f"Найденный минимум: {x_opt3}")
    print(f"Значение функции: {rosenbrock(x_opt3):.10f}")
    print(f"Итераций: {len(hist3)}")
    results.append(('Rosenbrock, constant step', hist3))
    
    # Градиентный спуск с адаптивным шагом
    print("\n2. Градиентный спуск с адаптивным шагом")
    x_opt4, hist4, traj4 = gradient_descent_adaptive(
        rosenbrock, rosenbrock_gradient, x0_rosen, max_iter=1000
    )
    print(f"Найденный минимум: {x_opt4}")
    print(f"Значение функции: {rosenbrock(x_opt4):.10f}")
    print(f"Итераций: {len(hist4)}")
    results.append(('Rosenbrock, adaptive step', hist4))
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    # Сравнение сходимости для функции сферы
    fig1, _ = plot_comparison(
        [hist1, hist2],
        ['Постоянный шаг', 'Адаптивный шаг'],
        'Сходимость для функции сферы'
    )
    save_plot(fig1, 'lab1_sphere_convergence.png', folder='lab_01/output')
    print("Сохранён график: lab_01/output/lab1_sphere_convergence.png")
    
    # Сравнение сходимости для функции Розенброка
    fig2, _ = plot_comparison(
        [hist3, hist4],
        ['Постоянный шаг', 'Адаптивный шаг'],
        'Сходимость для функции Розенброка'
    )
    save_plot(fig2, 'lab1_rosenbrock_convergence.png', folder='lab_01/output')
    print("Сохранён график: lab_01/output/lab1_rosenbrock_convergence.png")
    
    # Траектория на функции сферы
    fig3, _ = plot_trajectory(
        sphere, [-6, 6], [-6, 6], traj1,
        'Траектория градиентного спуска (постоянный шаг) - Сфера'
    )
    save_plot(fig3, 'lab1_sphere_trajectory.png', folder='lab_01/output')
    print("Сохранён график: lab_01/output/lab1_sphere_trajectory.png")
    
    # Траектория на функции Розенброка
    fig4, _ = plot_trajectory(
        rosenbrock, [-2, 3], [-1, 4], traj3[:100],  # Ограничим траекторию
        'Траектория градиентного спуска - Розенброк'
    )
    save_plot(fig4, 'lab1_rosenbrock_trajectory.png', folder='lab_01/output')
    print("Сохранён график: lab_01/output/lab1_rosenbrock_trajectory.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №1 завершена!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_lab1()
