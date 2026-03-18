"""
Модуль с функциями визуализации для лабораторных работ.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_function_2d(func, x_range, y_range, resolution=100, title="Функция"):
    """
    Построить 2D контурный график функции двух переменных.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cm.viridis)
    ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5)
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax


def plot_function_3d(func, x_range, y_range, resolution=100, title="Функция"):
    """
    Построить 3D поверхность функции двух переменных.
    """
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


def plot_convergence(history, title="Сходимость метода"):
    """
    Построить график сходимости оптимизационного метода.
    history: список значений функции на каждой итерации
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, linewidth=2)
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Значение функции')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    return fig, ax


def plot_comparison(histories, labels, title="Сравнение методов"):
    """
    Построить график сходимости нескольких методов.
    histories: список списков (история для каждого метода)
    labels: названия методов
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for history, label in zip(histories, labels):
        ax.plot(history, linewidth=2, label=label)
    
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Значение функции')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    return fig, ax


def plot_trajectory(func, x_range, y_range, trajectory, title="Траектория оптимизации"):
    """
    Построить траекторию оптимизации на контурном графике.
    trajectory: список точек (x, y) в процессе оптимизации
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(100):
        for j in range(100):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cm.viridis)
    ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5)
    
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', linewidth=2, 
            markersize=4, label='Траектория')
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, 
            label='Старт')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'yx', markersize=15, 
            label='Финиш')
    
    plt.colorbar(contour, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    return fig, ax


def save_plot(fig, filename, folder='output'):
    """Сохранить график в файл."""
    import os
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
