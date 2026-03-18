"""
Лабораторная работа №5
Стохастические методы оптимизации

Реализация:
1. Стохастический градиентный спуск (SGD)
2. Mini-batch градиентный спуск
3. SGD с различными стратегиями изменения learning rate
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import sphere, sphere_gradient
from utils.visualization import plot_convergence, plot_comparison, save_plot


# ========== Функции потерь для машинного обучения ==========

def mse_loss(X, y, w):
    """
    Функция потерь MSE (Mean Squared Error).
    L(w) = (1/2n) * ||Xw - y||²
    """
    n = len(y)
    predictions = X @ w
    return 0.5 * np.mean((predictions - y) ** 2)


def mse_gradient(X, y, w):
    """Градиент MSE."""
    n = len(y)
    predictions = X @ w
    return X.T @ (predictions - y) / n


def logistic_loss(X, y, w, lambda_reg=0.0):
    """
    Логистическая функция потерь с L2-регуляризацией.
    L(w) = -mean(y*log(sigmoid(Xw)) + (1-y)*log(1-sigmoid(Xw))) + (λ/2)*||w||²
    """
    n = len(y)
    z = X @ w
    # Численно стабильная версия
    loss = np.mean(np.log(1 + np.exp(-y * z))) + 0.5 * lambda_reg * np.sum(w ** 2)
    return loss


def logistic_gradient(X, y, w, lambda_reg=0.0):
    """Градиент логистической функции потерь."""
    n = len(y)
    z = X @ w
    sigmoid = 1 / (1 + np.exp(-y * z))
    grad = -X.T @ (y * (1 - sigmoid)) / n + lambda_reg * w
    return grad


# ========== Стохастические методы ==========

def sgd(f, grad_f, x0, X, y, lr=0.01, max_epochs=100, tol=1e-6, 
        random_state=None, verbose=False):
    """
    Стохастический градиентный спуск (SGD).
    
    На каждой итерации используется один случайный пример.
    
    Параметры:
        X: матрица объектов (n_samples, n_features)
        y: вектор целевых значений
        max_epochs: количество эпох (проходов по данным)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    x = np.array(x0, dtype=float)
    n_samples = len(y)
    history = []
    
    for epoch in range(max_epochs):
        # Перемешивание данных
        indices = np.random.permutation(n_samples)
        epoch_losses = []
        
        for i in indices:
            # Градиент по одному примеру
            X_i = X[i:i+1]
            y_i = y[i:i+1]
            grad = grad_f(X_i, y_i, x)
            
            # Обновление
            x = x - lr * grad
            epoch_losses.append(f(X_i, y_i, x))
        
        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)
        
        if verbose and epoch % 10 == 0:
            print(f"Эпоха {epoch}: loss = {avg_loss:.6f}")
        
        # Проверка сходимости
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            if verbose:
                print(f"Сходимость на эпохе {epoch}")
            break
    
    return x, history


def mini_batch_gd(f, grad_f, x0, X, y, lr=0.01, batch_size=32, 
                   max_epochs=100, tol=1e-6, random_state=None, verbose=False):
    """
    Mini-batch градиентный спуск.
    
    На каждой итерации используется батч из batch_size примеров.
    
    Параметры:
        batch_size: размер батча
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    x = np.array(x0, dtype=float)
    n_samples = len(y)
    history = []
    
    for epoch in range(max_epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            
            grad = grad_f(X_batch, y_batch, x)
            x = x - lr * grad
            
            epoch_losses.append(f(X_batch, y_batch, x))
        
        avg_loss = np.mean(epoch_losses)
        history.append(avg_loss)
        
        if verbose and epoch % 10 == 0:
            print(f"Эпоха {epoch}: loss = {avg_loss:.6f}")
        
        if len(history) > 1 and abs(history[-2] - history[-1]) < tol:
            break
    
    return x, history


def sgd_with_momentum(f, grad_f, x0, X, y, lr=0.01, beta=0.9, 
                       max_epochs=100, random_state=None, verbose=False):
    """
    SGD с инерцией (Momentum).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    n_samples = len(y)
    history = []
    
    for epoch in range(max_epochs):
        indices = np.random.permutation(n_samples)
        epoch_losses = []
        
        for i in indices:
            X_i = X[i:i+1]
            y_i = y[i:i+1]
            grad = grad_f(X_i, y_i, x)
            
            v = beta * v + lr * grad
            x = x - v
            
            epoch_losses.append(f(X_i, y_i, x))
        
        history.append(np.mean(epoch_losses))
    
    return x, history


# ========== Стратегии изменения learning rate ==========

def step_decay(initial_lr, drop_rate=0.5, epochs_drop=10):
    """
    Ступенчатое уменьшение lr.
    lr = initial_lr * drop_rate^(epoch // epochs_drop)
    """
    def scheduler(epoch):
        return initial_lr * (drop_rate ** (epoch // epochs_drop))
    return scheduler


def exponential_decay(initial_lr, decay_rate=0.95):
    """
    Экспоненциальное затухание lr.
    lr = initial_lr * decay_rate^epoch
    """
    def scheduler(epoch):
        return initial_lr * (decay_rate ** epoch)
    return scheduler


def adaptive_decay(initial_lr, decay_rate=0.01):
    """
    Адаптивное затухание.
    lr = initial_lr / (1 + decay_rate * epoch)
    """
    def scheduler(epoch):
        return initial_lr / (1 + decay_rate * epoch)
    return scheduler


def sgd_with_lr_schedule(f, grad_f, x0, X, y, lr_scheduler, max_epochs=100,
                          random_state=None, verbose=False):
    """
    SGD с расписанием изменения learning rate.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    x = np.array(x0, dtype=float)
    n_samples = len(y)
    history = []
    lr_history = []
    
    for epoch in range(max_epochs):
        lr = lr_scheduler(epoch)
        lr_history.append(lr)
        
        indices = np.random.permutation(n_samples)
        epoch_losses = []
        
        for i in indices:
            X_i = X[i:i+1]
            y_i = y[i:i+1]
            grad = grad_f(X_i, y_i, x)
            x = x - lr * grad
            
            epoch_losses.append(f(X_i, y_i, x))
        
        history.append(np.mean(epoch_losses))
        
        if verbose and epoch % 10 == 0:
            print(f"Эпоха {epoch}: lr = {lr:.6f}, loss = {history[-1]:.6f}")
    
    return x, history, lr_history


def run_lab5():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №5: Стохастические методы оптимизации")
    print("=" * 60)
    
    # ========== Генерация данных ==========
    print("\n--- Генерация данных ---")
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Данные для регрессии
    X_reg = np.random.randn(n_samples, n_features)
    true_w = np.random.randn(n_features)
    y_reg = X_reg @ true_w + 0.1 * np.random.randn(n_samples)
    
    # Данные для классификации
    X_clf = np.random.randn(n_samples, n_features)
    true_w_clf = np.random.randn(n_features)
    y_clf = np.sign(X_clf @ true_w_clf + 0.5 * np.random.randn(n_samples))
    
    x0 = np.zeros(n_features)
    
    # ========== Тест 1: Сравнение SGD и Mini-batch ==========
    print("\n--- Тест 1: Сравнение SGD и Mini-batch GD ---")
    
    _, history_sgd = sgd(
        mse_loss, mse_gradient, x0, X_reg, y_reg,
        lr=0.01, max_epochs=50, random_state=42
    )
    
    _, history_mb = mini_batch_gd(
        mse_loss, mse_gradient, x0, X_reg, y_reg,
        lr=0.01, batch_size=32, max_epochs=50, random_state=42
    )
    
    _, history_mb_large = mini_batch_gd(
        mse_loss, mse_gradient, x0, X_reg, y_reg,
        lr=0.01, batch_size=128, max_epochs=50, random_state=42
    )
    
    print(f"SGD: final loss = {history_sgd[-1]:.6f}")
    print(f"Mini-batch (32): final loss = {history_mb[-1]:.6f}")
    print(f"Mini-batch (128): final loss = {history_mb_large[-1]:.6f}")
    
    # ========== Тест 2: Стратегии изменения lr ==========
    print("\n--- Тест 2: Стратегии изменения learning rate ---")
    
    strategies = {
        'Constant': lambda e: 0.01,
        'Step Decay': step_decay(0.1, drop_rate=0.5, epochs_drop=10),
        'Exponential': exponential_decay(0.1, decay_rate=0.95),
        'Adaptive': adaptive_decay(0.1, decay_rate=0.01),
    }
    
    histories_lr = {}
    
    for name, scheduler in strategies.items():
        _, history, _ = sgd_with_lr_schedule(
            mse_loss, mse_gradient, x0, X_reg, y_reg,
            scheduler, max_epochs=50, random_state=42
        )
        histories_lr[name] = history
        print(f"{name:15s}: final loss = {history[-1]:.6f}")
    
    # ========== Тест 3: Логистическая регрессия ==========
    print("\n--- Тест 3: Логистическая регрессия ---")
    
    _, history_log_sgd = sgd(
        logistic_loss, logistic_gradient, x0, X_clf, y_clf,
        lr=0.1, max_epochs=50, random_state=42
    )
    
    _, history_log_mb = mini_batch_gd(
        logistic_loss, logistic_gradient, x0, X_clf, y_clf,
        lr=0.1, batch_size=32, max_epochs=50, random_state=42
    )
    
    print(f"SGD Logistic: final loss = {history_log_sgd[-1]:.6f}")
    print(f"Mini-batch Logistic: final loss = {history_log_mb[-1]:.6f}")
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    # Сравнение SGD и Mini-batch
    fig1, _ = plot_comparison(
        [history_sgd, history_mb, history_mb_large],
        ['SGD', 'Mini-batch (32)', 'Mini-batch (128)'],
        'Сравнение SGD и Mini-batch GD'
    )
    save_plot(fig1, 'lab5_sgd_vs_mb.png', folder='lab_05/output')
    print("Сохранён график: lab_05/output/lab5_sgd_vs_mb.png")
    
    # Стратегии изменения lr
    fig2, _ = plot_comparison(
        list(histories_lr.values()),
        list(histories_lr.keys()),
        'Стратегии изменения learning rate'
    )
    save_plot(fig2, 'lab5_lr_strategies.png', folder='lab_05/output')
    print("Сохранён график: lab_05/output/lab5_lr_strategies.png")
    
    # Логистическая регрессия
    fig3, _ = plot_comparison(
        [history_log_sgd, history_log_mb],
        ['SGD', 'Mini-batch'],
        'Обучение логистической регрессии'
    )
    save_plot(fig3, 'lab5_logistic.png', folder='lab_05/output')
    print("Сохранён график: lab_05/output/lab5_logistic.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №5 завершена!")
    print("=" * 60)


if __name__ == "__main__":
    run_lab5()
