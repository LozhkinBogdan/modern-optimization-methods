"""
Лабораторная работа №2
Методы ускорения градиентного спуска

Реализация:
1. Momentum (градиентный спуск с инерцией)
2. Nesterov Accelerated Gradient (NAG)
3. AdaGrad
4. Adam
5. RMSprop
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import sphere, sphere_gradient, rosenbrock, rosenbrock_gradient, rastrigin, rastrigin_gradient
from utils.visualization import plot_convergence, plot_comparison, save_plot


def momentum(f, grad_f, x0, lr=0.01, beta=0.9, max_iter=1000, tol=1e-6):
    """
    Градиентный спуск с Momentum (инерцией).
    
    v_{k+1} = beta * v_k + lr * grad_f(x_k)
    x_{k+1} = x_k - v_{k+1}
    
    Параметры:
        beta: коэффициент инерции (обычно 0.9)
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        v = beta * v + lr * grad
        x = x - v
        
        history.append(f(x))
    
    return x, history


def nesterov(f, grad_f, x0, lr=0.01, beta=0.9, max_iter=1000, tol=1e-6):
    """
    Nesterov Accelerated Gradient (NAG).
    
    Ключевое отличие от Momentum: градиент вычисляется в точке с учётом инерции.
    
    v_{k+1} = beta * v_k + lr * grad_f(x_k - beta * v_k)
    x_{k+1} = x_k - v_{k+1}
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    history = [f(x)]
    
    for i in range(max_iter):
        # Градиент вычисляется в точке "с заглядыванием вперёд"
        x_ahead = x - beta * v
        grad = grad_f(x_ahead)
        
        if np.linalg.norm(grad) < tol:
            break
        
        v = beta * v + lr * grad
        x = x - v
        
        history.append(f(x))
    
    return x, history


def adagrad(f, grad_f, x0, lr=0.1, epsilon=1e-8, max_iter=1000, tol=1e-6):
    """
    AdaGrad (Adaptive Gradient).
    
    Адаптивно изменяет learning rate для каждого параметра.
    
    G_{k+1} = G_k + grad_f(x_k)^2
    x_{k+1} = x_k - lr * grad_f(x_k) / (sqrt(G_{k+1}) + epsilon)
    """
    x = np.array(x0, dtype=float)
    G = np.zeros_like(x)
    history = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        G = G + grad ** 2
        x = x - lr * grad / (np.sqrt(G) + epsilon)
        
        history.append(f(x))
    
    return x, history


def rmsprop(f, grad_f, x0, lr=0.01, beta=0.9, epsilon=1e-8, max_iter=1000, tol=1e-6):
    """
    RMSprop (Root Mean Square Propagation).
    
    Модификация AdaGrad с экспоненциальным затуханием накопленных градиентов.
    
    E[g^2]_{k+1} = beta * E[g^2]_k + (1 - beta) * grad_f(x_k)^2
    x_{k+1} = x_k - lr * grad_f(x_k) / (sqrt(E[g^2]_{k+1}) + epsilon)
    """
    x = np.array(x0, dtype=float)
    E_g2 = np.zeros_like(x)
    history = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        E_g2 = beta * E_g2 + (1 - beta) * grad ** 2
        x = x - lr * grad / (np.sqrt(E_g2) + epsilon)
        
        history.append(f(x))
    
    return x, history


def adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, 
         max_iter=1000, tol=1e-6):
    """
    Adam (Adaptive Moment Estimation).
    
    Комбинирует идеи Momentum и RMSprop.
    
    m_{k+1} = beta1 * m_k + (1 - beta1) * grad_f(x_k)  # первый момент
    v_{k+1} = beta2 * v_k + (1 - beta2) * grad_f(x_k)^2  # второй момент
    m_hat = m_{k+1} / (1 - beta1^{k+1})  # коррекция смещения
    v_hat = v_{k+1} / (1 - beta2^{k+1})
    x_{k+1} = x_k - lr * m_hat / (sqrt(v_hat) + epsilon)
    """
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = [f(x)]
    
    for k in range(max_iter):
        grad = grad_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        
        # Коррекция смещения
        m_hat = m / (1 - beta1 ** (k + 1))
        v_hat = v / (1 - beta2 ** (k + 1))
        
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        history.append(f(x))
    
    return x, history


def run_lab2():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №2: Методы ускорения градиентного спуска")
    print("=" * 60)
    
    # Параметры для тестирования
    x0_sphere = np.array([5.0, 5.0])
    x0_rosen = np.array([2.0, 2.0])
    x0_rastrigin = np.array([2.5, 2.5])
    
    # Методы для сравнения
    methods = {
        'Momentum': lambda f, g, x0: momentum(f, g, x0, lr=0.1, beta=0.9, max_iter=200),
        'Nesterov': lambda f, g, x0: nesterov(f, g, x0, lr=0.1, beta=0.9, max_iter=200),
        'AdaGrad': lambda f, g, x0: adagrad(f, g, x0, lr=0.5, max_iter=200),
        'RMSprop': lambda f, g, x0: rmsprop(f, g, x0, lr=0.05, beta=0.9, max_iter=200),
        'Adam': lambda f, g, x0: adam(f, g, x0, lr=0.1, beta1=0.9, beta2=0.999, max_iter=200),
    }
    
    all_results = {}
    
    # ========== Тест 1: Функция сферы ==========
    print("\n--- Тест 1: Функция сферы ---")
    histories_sphere = []
    labels = []
    
    for name, method in methods.items():
        x_opt, history = method(sphere, sphere_gradient, x0_sphere)
        print(f"{name:12s}: f(x) = {sphere(x_opt):.2e}, итераций: {len(history)}")
        histories_sphere.append(history)
        labels.append(name)
    
    all_results['sphere'] = (histories_sphere, labels)
    
    # ========== Тест 2: Функция Розенброка ==========
    print("\n--- Тест 2: Функция Розенброка ---")
    histories_rosen = []
    
    for name, method in methods.items():
        x_opt, history = method(rosenbrock, rosenbrock_gradient, x0_rosen)
        print(f"{name:12s}: f(x) = {rosenbrock(x_opt):.2e}, итераций: {len(history)}")
        histories_rosen.append(history)
    
    all_results['rosenbrock'] = (histories_rosen, labels)
    
    # ========== Тест 3: Функция Растригина ==========
    print("\n--- Тест 3: Функция Растригина ---")
    histories_rastrigin = []
    
    for name, method in methods.items():
        x_opt, history = method(rastrigin, rastrigin_gradient, x0_rastrigin)
        print(f"{name:12s}: f(x) = {rastrigin(x_opt):.2e}, итераций: {len(history)}")
        histories_rastrigin.append(history)
    
    all_results['rastrigin'] = (histories_rastrigin, labels)
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    # Сравнение для сферы
    fig1, _ = plot_comparison(
        histories_sphere, labels,
        'Сравнение методов ускорения - Функция сферы'
    )
    save_plot(fig1, 'lab2_sphere_comparison.png', folder='lab_02/output')
    print("Сохранён график: lab_02/output/lab2_sphere_comparison.png")
    
    # Сравнение для Розенброка
    fig2, _ = plot_comparison(
        histories_rosen, labels,
        'Сравнение методов ускорения - Функция Розенброка'
    )
    save_plot(fig2, 'lab2_rosenbrock_comparison.png', folder='lab_02/output')
    print("Сохранён график: lab_02/output/lab2_rosenbrock_comparison.png")
    
    # Сравнение для Растригина
    fig3, _ = plot_comparison(
        histories_rastrigin, labels,
        'Сравнение методов ускорения - Функция Растригина'
    )
    save_plot(fig3, 'lab2_rastrigin_comparison.png', folder='lab_02/output')
    print("Сохранён график: lab_02/output/lab2_rastrigin_comparison.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №2 завершена!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_lab2()
