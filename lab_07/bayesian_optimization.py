"""
Лабораторная работа №7
Байесовская оптимизация

Реализация:
1. Гауссовские процессы (регрессия)
2. Функции приобретения: EI, PI, UCB
3. Байесовская оптимизация
4. Оптимизация гиперпараметров модели ML
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_convergence, save_plot
import matplotlib.pyplot as plt


# ========== Гауссовский процесс ==========

class GaussianProcessRegressor:
    """
    Гауссовский процесс для регрессии.
    
    Использует RBF ядро (квадратичная экспонента).
    """
    
    def __init__(self, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6):
        """
        Параметры:
            length_scale: параметр длины корреляции RBF ядра
            sigma_f: амплитуда функции
            sigma_n: шум наблюдений
        """
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        
        self.X_train = None
        self.y_train = None
        self.K = None
        self.K_inv = None
        self.alpha = None
    
    def _rbf_kernel(self, X1, X2):
        """
        RBF ядро (квадратичная экспонента).
        K(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * length_scale^2))
        """
        sq_dist = cdist(X1, X2, 'sqeuclidean')
        return self.sigma_f ** 2 * np.exp(-sq_dist / (2 * self.length_scale ** 2))
    
    def fit(self, X, y):
        """
        Обучение гауссовского процесса.
        """
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y)
        
        # Ковариационная матрица
        self.K = self._rbf_kernel(self.X_train, self.X_train)
        self.K += self.sigma_n ** 2 * np.eye(len(self.y_train))
        
        # Обратная матрица (для эффективности)
        self.K_inv = np.linalg.inv(self.K)
        
        # Коэффициенты для предсказания
        self.alpha = self.K_inv @ self.y_train
        
        return self
    
    def predict(self, X_test, return_std=False):
        """
        Предсказание с оценкой неопределённости.
        """
        X_test = np.atleast_2d(X_test)
        
        # Ковариация между тренировочными и тестовыми точками
        K_s = self._rbf_kernel(self.X_train, X_test)
        
        # Среднее
        mu = K_s.T @ self.alpha
        
        if return_std:
            # Ковариация тестовых точек
            K_ss = self._rbf_kernel(X_test, X_test)
            
            # Дисперсия
            var = np.diag(K_ss) - np.diag(K_s.T @ self.K_inv @ K_s)
            var = np.maximum(var, 1e-10)  # Численная стабильность
            std = np.sqrt(var)
            
            return mu, std
        
        return mu
    
    def predict_with_uncertainty(self, X_test):
        """Предсказание с возвращением стандартного отклонения."""
        return self.predict(X_test, return_std=True)


# ========== Функции приобретения ==========

def expected_improvement(x, gp, y_best, xi=0.01):
    """
    Expected Improvement (EI).
    
    EI(x) = E[max(f(x) - y_best - xi, 0)]
    
    Для гауссовского процесса имеет аналитическую форму:
    EI(x) = (mu - y_best - xi) * Phi(z) + sigma * phi(z)
    где z = (mu - y_best - xi) / sigma
    """
    x = np.atleast_2d(x)
    mu, sigma = gp.predict(x, return_std=True)
    
    # Избегаем деления на ноль
    sigma = np.maximum(sigma, 1e-10)
    
    z = (mu - y_best - xi) / sigma
    
    # Phi(z) - CDF стандартного нормального распределения
    from scipy.stats import norm
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    ei = (mu - y_best - xi) * Phi_z + sigma * phi_z
    
    return ei.item() if len(ei) == 1 else ei.flatten()


def probability_of_improvement(x, gp, y_best, xi=0.01):
    """
    Probability of Improvement (PI).
    
    PI(x) = P(f(x) > y_best + xi) = Phi((mu - y_best - xi) / sigma)
    """
    x = np.atleast_2d(x)
    mu, sigma = gp.predict(x, return_std=True)
    
    sigma = np.maximum(sigma, 1e-10)
    
    from scipy.stats import norm
    z = (mu - y_best - xi) / sigma
    pi = norm.cdf(z)
    
    return pi.item() if len(pi) == 1 else pi.flatten()


def upper_confidence_bound(x, gp, kappa=2.576):
    """
    Upper Confidence Bound (UCB).
    
    UCB(x) = mu(x) + kappa * sigma(x)
    
    kappa управляет балансом exploration/exploitation:
    - kappa > 2.576: больше exploration (99% доверительный интервал)
    - kappa = 1.96: 95% доверительный интервал
    """
    x = np.atleast_2d(x)
    mu, sigma = gp.predict(x, return_std=True)
    
    ucb = mu + kappa * sigma
    
    return ucb.item() if len(ucb) == 1 else ucb.flatten()


# ========== Байесовская оптимизация ==========

class BayesianOptimization:
    """
    Байесовская оптимизация чёрного ящика.
    """
    
    def __init__(self, f, bounds, gp_params=None, acquisition='ei'):
        """
        Параметры:
            f: целевая функция
            bounds: границы поиска [(low, high), ...]
            gp_params: параметры гауссовского процесса
            acquisition: функция приобретения ('ei', 'pi', 'ucb')
        """
        self.f = f
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        
        # Параметры GP
        if gp_params is None:
            gp_params = {'length_scale': 1.0, 'sigma_f': 1.0, 'sigma_n': 1e-6}
        self.gp_params = gp_params
        
        # Функция приобретения
        self.acquisition = acquisition.lower()
        
        # Данные
        self.X_samples = []
        self.y_samples = []
        
        # Лучшее значение
        self.y_best = float('inf')
        self.x_best = None
    
    def _acquisition_function(self, x, gp):
        """Вычисление функции приобретения."""
        if self.acquisition == 'ei':
            return -expected_improvement(x, gp, self.y_best)
        elif self.acquisition == 'pi':
            return -probability_of_improvement(x, gp, self.y_best)
        elif self.acquisition == 'ucb':
            return -upper_confidence_bound(x, gp)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
    
    def _suggest_next_point(self):
        """Предложить следующую точку для оценки."""
        if len(self.X_samples) == 0:
            # Случайная точка для первой итерации
            return np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1]
            )
        
        # Обучение GP
        gp = GaussianProcessRegressor(**self.gp_params)
        gp.fit(np.array(self.X_samples), np.array(self.y_samples))
        
        # Оптимизация функции приобретения
        result = minimize(
            lambda x: self._acquisition_function(x, gp),
            x0=np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]),
            method='L-BFGS-B',
            bounds=self.bounds,
            options={'maxiter': 100}
        )
        
        return result.x
    
    def optimize(self, n_iterations=20, n_initial=3, random_state=None):
        """
        Запуск байесовской оптимизации.
        
        Параметры:
            n_iterations: количество итераций
            n_initial: количество начальных случайных точек
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Начальные точки
        for _ in range(n_initial):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = self.f(x)
            self.X_samples.append(x)
            self.y_samples.append(y)
            
            if y < self.y_best:
                self.y_best = y
                self.x_best = x.copy()
        
        history = [(self.x_best.copy(), self.y_best)]
        
        # Итерации
        for i in range(n_iterations):
            # Предложить следующую точку
            x_next = self._suggest_next_point()
            
            # Оценить функцию
            y_next = self.f(x_next)
            
            # Добавить в данные
            self.X_samples.append(x_next)
            self.y_samples.append(y_next)
            
            # Обновить лучшее
            if y_next < self.y_best:
                self.y_best = y_next
                self.x_best = x_next.copy()
            
            history.append((self.x_best.copy(), self.y_best))
            
            print(f"Итерация {i + 1}/{n_iterations}: f(x) = {y_next:.6f}, best = {self.y_best:.6f}")
        
        return self.x_best, self.y_best, history
    
    def get_history(self):
        """Получить историю оптимизации."""
        return [h[1] for h in history] if hasattr(self, 'history') else []


def run_lab7():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №7: Байесовская оптимизация")
    print("=" * 60)
    
    # ========== Тестовая функция ==========
    print("\n--- Тестовая функция: 1D функция Бранда ---")
    
    def branin_1d(x):
        """Упрощённая 1D тестовая функция."""
        x = x[0] if len(x.shape) > 0 else x
        return (x - 2) ** 2 + 0.5 * np.sin(5 * x) + 3
    
    bounds_1d = [(0, 6)]
    
    # Визуализация функции
    x_test = np.linspace(0, 6, 100).reshape(-1, 1)
    y_test = np.array([branin_1d(x) for x in x_test])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_test.flatten(), y_test, 'b-', linewidth=2, label='Функция')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Тестовая функция')
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'lab7_test_function.png', folder='lab_07/output')
    print("Сохранён график: lab_07/output/lab7_test_function.png")
    
    # ========== Оптимизация с разными функциями приобретения ==========
    print("\n--- Сравнение функций приобретения ---")
    
    acquisitions = ['ei', 'pi', 'ucb']
    histories = {}
    
    for acq in acquisitions:
        print(f"\n--- Acquisition: {acq.upper()} ---")
        
        bo = BayesianOptimization(
            branin_1d, bounds_1d,
            acquisition=acq,
            gp_params={'length_scale': 1.0, 'sigma_f': 1.0, 'sigma_n': 1e-6}
        )
        
        x_opt, y_opt, history = bo.optimize(n_iterations=15, n_initial=3, random_state=42)
        
        histories[acq] = [h[1] for h in history]
        
        print(f"Найденный минимум: f({x_opt[0]:.4f}) = {y_opt:.6f}")
    
    # ========== 2D функция ==========
    print("\n--- 2D функция (Растригина) ---")
    
    def rastrigin_2d(x):
        return x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])) + 20
    
    bounds_2d = [(-3, 3), (-3, 3)]
    
    bo_2d = BayesianOptimization(
        rastrigin_2d, bounds_2d,
        acquisition='ei',
        gp_params={'length_scale': 1.5, 'sigma_f': 1.0, 'sigma_n': 1e-6}
    )
    
    x_opt_2d, y_opt_2d, history_2d = bo_2d.optimize(n_iterations=20, n_initial=5, random_state=42)
    
    print(f"Найденный минимум: f({x_opt_2d[0]:.4f}, {x_opt_2d[1]:.4f}) = {y_opt_2d:.6f}")
    
    histories['rastrigin_2d'] = [h[1] for h in history_2d]
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    # Сравнение функций приобретения
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for acq in acquisitions:
        ax.plot(histories[acq], 'o-', linewidth=2, label=f'{acq.upper()}')
    
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Лучшее значение')
    ax.set_title('Сравнение функций приобретения')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'lab7_acquisition_comparison.png', folder='lab_07/output')
    print("Сохранён график: lab_07/output/lab7_acquisition_comparison.png")
    
    # История для 2D
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(histories['rastrigin_2d'], 'o-', linewidth=2)
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Лучшее значение')
    ax.set_title('Байесовская оптимизация - 2D Растригина')
    ax.grid(True, alpha=0.3)
    save_plot(fig, 'lab7_rastrigin_2d.png', folder='lab_07/output')
    print("Сохранён график: lab_07/output/lab7_rastrigin_2d.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №7 завершена!")
    print("=" * 60)


if __name__ == "__main__":
    run_lab7()
