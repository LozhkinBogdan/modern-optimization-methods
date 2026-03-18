"""
Лабораторная работа №8
Применение методов оптимизации в машинном обучении

Реализация:
1. Линейная регрессия с различными методами оптимизации
2. Логистическая регрессия
3. Нейронная сеть (1 скрытый слой)
4. Сравнение оптимизаторов на реальных данных
"""

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_convergence, plot_comparison, save_plot
import matplotlib.pyplot as plt


# ========== Активационные функции ==========

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    # Численно стабильная версия
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# ========== Оптимизаторы ==========

class SGDOptimizer:
    """Стохастический градиентный спуск."""
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]


class MomentumOptimizer:
    """SGD с Momentum."""
    
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = None
    
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        self.velocity = [
            self.beta * v + self.lr * g
            for v, g in zip(self.velocity, grads)
        ]
        
        return [p - v for p, v in zip(params, self.velocity)]


class AdamOptimizer:
    """Adam оптимизатор."""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]
        
        # Коррекция смещения
        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]
        
        return [
            p - self.lr * mh / (np.sqrt(vh) + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]


class AdaGradOptimizer:
    """AdaGrad оптимизатор."""
    
    def __init__(self, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.G = None
    
    def update(self, params, grads):
        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]
        
        self.G = [g + grad ** 2 for g, grad in zip(self.G, grads)]
        
        return [
            p - self.lr * grad / (np.sqrt(g) + self.epsilon)
            for p, grad, g in zip(params, grads, self.G)
        ]


# ========== Модели ==========

class LinearRegression:
    """Линейная регрессия."""
    
    def __init__(self):
        self.w = None
        self.b = None
        self.history = []
    
    def fit(self, X, y, optimizer, n_iterations=1000, tol=1e-6):
        """
        Обучение линейной регрессии.
        MSE Loss: L = (1/2n) * ||Xw + b - y||^2
        """
        n_samples, n_features = X.shape
        
        # Инициализация
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        self.history = []
        
        prev_loss = float('inf')
        
        for i in range(n_iterations):
            # Forward pass
            y_pred = X @ self.w + self.b
            
            # Loss
            loss = 0.5 * np.mean((y_pred - y) ** 2)
            self.history.append(loss)
            
            # Проверка сходимости
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
            
            # Градиенты
            dw = X.T @ (y_pred - y) / n_samples
            db = np.mean(y_pred - y)
            
            # Обновление
            self.w, self.b = optimizer.update([self.w, self.b], [dw, db])
        
        return self
    
    def predict(self, X):
        return X @ self.w + self.b
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


class LogisticRegression:
    """Логистическая регрессия."""
    
    def __init__(self):
        self.w = None
        self.b = None
        self.history = []
    
    def fit(self, X, y, optimizer, n_iterations=1000, tol=1e-6):
        """
        Обучение логистической регрессии.
        Binary Cross-Entropy: L = -mean(y*log(p) + (1-y)*log(1-p))
        """
        n_samples, n_features = X.shape
        
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        self.history = []
        
        prev_loss = float('inf')
        
        for i in range(n_iterations):
            # Forward pass
            z = X @ self.w + self.b
            y_pred = sigmoid(z)
            
            # Loss (с добавлением epsilon для численной стабильности)
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            self.history.append(loss)
            
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
            
            # Градиенты
            dz = y_pred - y
            dw = X.T @ dz / n_samples
            db = np.mean(dz)
            
            # Обновление
            self.w, self.b = optimizer.update([self.w, self.b], [dw, db])
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class NeuralNetwork:
    """
    Нейронная сеть с одним скрытым слоем.
    
    Архитектура:
    Input -> Hidden (ReLU) -> Output (sigmoid для классификации)
    """
    
    def __init__(self, n_input, n_hidden=32, n_output=1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        # Инициализация весов (Xavier)
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_output))
        
        self.history = []
    
    def _forward(self, X):
        """Прямой проход."""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def _backward(self, X, y):
        """Обратный проход (backpropagation)."""
        n_samples = X.shape[0]
        
        # Выходной слой
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = self.a1.T @ dz2 / n_samples
        db2 = np.mean(dz2, axis=0, keepdims=True)
        
        # Скрытый слой
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = X.T @ dz1 / n_samples
        db1 = np.mean(dz1, axis=0)
        
        return [dW1, db1, dW2, db2]
    
    def fit(self, X, y, optimizer, n_iterations=1000, tol=1e-6):
        """Обучение нейронной сети."""
        self.history = []
        prev_loss = float('inf')
        
        for i in range(n_iterations):
            # Forward
            y_pred = self._forward(X)
            
            # Loss
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.mean(y.reshape(-1, 1) * np.log(y_pred_clipped) +
                           (1 - y.reshape(-1, 1)) * np.log(1 - y_pred_clipped))
            self.history.append(loss)
            
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
            
            # Backward
            grads = self._backward(X, y)
            
            # Обновление
            params = [self.W1, self.b1, self.W2, self.b2]
            new_params = optimizer.update(params, grads)
            self.W1, self.b1, self.W2, self.b2 = new_params
        
        return self
    
    def predict_proba(self, X):
        return self._forward(X).flatten()
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def run_lab8():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №8: Применение в машинном обучении")
    print("=" * 60)
    
    # ========== Генерация данных ==========
    print("\n--- Генерация данных ---")
    
    # Данные для регрессии
    X_reg, y_reg = make_regression(n_samples=500, n_features=10, 
                                    n_informative=5, noise=10, random_state=42)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    #Scaler для регрессии
    scaler_reg = StandardScaler()
    X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
    X_reg_test_scaled = scaler_reg.transform(X_reg_test)
    
    # Данные для классификации
    X_clf, y_clf = make_classification(n_samples=500, n_features=10,
                                        n_informative=5, n_redundant=2,
                                        random_state=42)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Scaler для классификации
    scaler_clf = StandardScaler()
    X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
    X_clf_test_scaled = scaler_clf.transform(X_clf_test)
    
    print(f"Регрессия: {X_reg_train.shape[0]} train, {X_reg_test.shape[0]} test")
    print(f"Классификация: {X_clf_train.shape[0]} train, {X_clf_test.shape[0]} test")
    
    # ========== Оптимизаторы ==========
    # Определяем классы и параметры, создаём экземпляры для каждого теста
    optimizer_configs = {
        'SGD': {'class': SGDOptimizer, 'params': {'lr': 0.1}},
        'Momentum': {'class': MomentumOptimizer, 'params': {'lr': 0.05, 'beta': 0.9}},
        'Adam': {'class': AdamOptimizer, 'params': {'lr': 0.1}},
        'AdaGrad': {'class': AdaGradOptimizer, 'params': {'lr': 0.5}},
    }

    # ========== Тест 1: Линейная регрессия ==========
    print("\n--- Тест 1: Линейная регрессия ---")

    reg_results = {}

    for name, config in optimizer_configs.items():
        opt = config['class'](**config['params'])
        model = LinearRegression()
        model.fit(X_reg_train_scaled, y_reg_train, optimizer=opt, n_iterations=500)
        
        train_score = model.score(X_reg_train_scaled, y_reg_train)
        test_score = model.score(X_reg_test_scaled, y_reg_test)

        reg_results[name] = model.history

        print(f"{name:12s}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

    # ========== Тест 2: Логистическая регрессия ==========
    print("\n--- Тест 2: Логистическая регрессия ---")

    clf_results = {}

    for name, config in optimizer_configs.items():
        opt = config['class'](**config['params'])
        model = LogisticRegression()
        model.fit(X_clf_train_scaled, y_clf_train, optimizer=opt, n_iterations=500)

        train_score = model.score(X_clf_train_scaled, y_clf_train)
        test_score = model.score(X_clf_test_scaled, y_clf_test)

        clf_results[name] = model.history

        print(f"{name:12s}: Train Acc = {train_score:.4f}, Test Acc = {test_score:.4f}")

    # ========== Тест 3: Нейронная сеть ==========
    print("\n--- Тест 3: Нейронная сеть (1 скрытый слой) ---")

    nn_results = {}

    for name, config in optimizer_configs.items():
        opt = config['class'](**config['params'])
        model = NeuralNetwork(n_input=10, n_hidden=32, n_output=1)
        model.fit(X_clf_train_scaled, y_clf_train, optimizer=opt, n_iterations=500)

        train_score = model.score(X_clf_train_scaled, y_clf_train)
        test_score = model.score(X_clf_test_scaled, y_clf_test)

        nn_results[name] = model.history

        print(f"{name:12s}: Train Acc = {train_score:.4f}, Test Acc = {test_score:.4f}")
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    # Линейная регрессия
    fig1, _ = plot_comparison(
        list(reg_results.values()),
        list(reg_results.keys()),
        'Линейная регрессия: Сравнение оптимизаторов'
    )
    save_plot(fig1, 'lab8_linear_regression.png', folder='lab_08/output')
    print("Сохранён график: lab_08/output/lab8_linear_regression.png")
    
    # Логистическая регрессия
    fig2, _ = plot_comparison(
        list(clf_results.values()),
        list(clf_results.keys()),
        'Логистическая регрессия: Сравнение оптимизаторов'
    )
    save_plot(fig2, 'lab8_logistic_regression.png', folder='lab_08/output')
    print("Сохранён график: lab_08/output/lab8_logistic_regression.png")
    
 # Нейронная сеть
    fig3, _ = plot_comparison(
        list(nn_results.values()),
        list(nn_results.keys()),
        'Нейронная сеть: Сравнение оптимизаторов'
    )
    save_plot(fig3, 'lab8_neural_network.png', folder='lab_08/output')
    print("Сохранён график: lab_08/output/lab8_neural_network.png")
    
    # Сравнение моделей для Adam
    fig4, _ = plot_comparison(
        [reg_results['Adam'], clf_results['Adam'], nn_results['Adam']],
        ['Linear Regression', 'Logistic Regression', 'Neural Network'],
        'Сравнение моделей (Adam optimizer)'
    )
    save_plot(fig4, 'lab8_model_comparison.png', folder='lab_08/output')
    print("Сохранён график: lab_08/output/lab8_model_comparison.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №8 завершена!")
    print("=" * 60)


if __name__ == "__main__":
    run_lab8()
