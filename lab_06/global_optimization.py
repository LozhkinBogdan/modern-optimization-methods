"""
Лабораторная работа №6
Методы глобальной оптимизации

Реализация:
1. Метод случайного поиска
2. Метод имитации отжига (Simulated Annealing)
3. Генетический алгоритм
4. Метод роя частиц (Particle Swarm Optimization)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.functions import sphere, rosenbrock, rastrigin, ackley
from utils.visualization import plot_convergence, plot_comparison, save_plot


# ========== Метод случайного поиска ==========

def random_search(f, bounds, n_iterations=1000, n_samples=100, 
                  random_state=None):
    """
    Метод случайного поиска (Random Search).
    
    Алгоритм:
    1. Генерируем n_samples случайных точек в области поиска
    2. Выбираем лучшую
    3. Повторяем, сужая область вокруг лучшей точки
    
    Параметры:
        bounds: список кортежей [(low_1, high_1), (low_2, high_2), ...]
        n_iterations: количество итераций
        n_samples: количество случайных точек на итерации
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_dims = len(bounds)
    bounds = np.array(bounds)
    
    # Лучшее решение
    best_x = None
    best_f = float('inf')
    
    # Текущий центр поиска
    center = np.mean(bounds, axis=1)
    # Текущий радиус поиска
    radius = (bounds[:, 1] - bounds[:, 0]) / 2
    
    history = []
    
    for iteration in range(n_iterations):
        # Генерация случайных точек в текущей области
        samples = np.random.uniform(
            center - radius,
            center + radius,
            size=(n_samples, n_dims)
        )
        
        # Оценка точек
        for i in range(n_samples):
            x = samples[i]
            # Проверка границ
            x = np.clip(x, bounds[:, 0], bounds[:, 1])
            fx = f(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        history.append(best_f)
        
        # Сужение области поиска
        if best_x is not None:
            center = best_x
        radius *= 0.95  # Уменьшение радиуса
    
    return best_x, best_f, history


# ========== Метод имитации отжига ==========

def simulated_annealing(f, x0, bounds, T0=1000, T_min=1e-8, 
                         alpha=0.99, max_iter=1000, random_state=None):
    """
    Метод имитации отжига (Simulated Annealing).
    
    Алгоритм:
    1. Генерируем соседнее решение
    2. Принимаем лучшее решение всегда
    3. Принимаем худшее решение с вероятностью exp(-ΔE / T)
    4. Уменьшаем температуру
    
    Параметры:
        T0: начальная температура
        T_min: минимальная температура
        alpha: коэффициент охлаждения (0 < alpha < 1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bounds = np.array(bounds)
    n_dims = len(bounds)
    
    x_current = np.array(x0, dtype=float)
    f_current = f(x_current)
    
    x_best = x_current.copy()
    f_best = f_current
    
    history = []
    T = T0
    
    for iteration in range(max_iter):
        # Генерация соседнего решения
        step_size = (bounds[:, 1] - bounds[:, 0]) * 0.01 * T / T0
        x_new = x_current + np.random.normal(0, step_size, n_dims)
        
        # Проверка границ
        x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])
        
        f_new = f(x_new)
        delta = f_new - f_current
        
        # Принятие решения
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x_current = x_new
            f_current = f_new
            
            if f_current < f_best:
                x_best = x_current.copy()
                f_best = f_current
        
        history.append(f_best)
        
        # Охлаждение
        T = T * alpha
        
        if T < T_min:
            break
    
    return x_best, f_best, history


# ========== Генетический алгоритм ==========

def genetic_algorithm(f, bounds, population_size=50, n_generations=100,
                       mutation_rate=0.1, crossover_rate=0.8,
                       elite_size=5, random_state=None):
    """
    Генетический алгоритм.
    
    Алгоритм:
    1. Инициализация популяции
    2. Оценка приспособленности
    3. Селекция (турнирная)
    4. Кроссовер
    5. Мутация
    6. Повторять
    
    Параметры:
        population_size: размер популяции
        n_generations: количество поколений
        mutation_rate: вероятность мутации
        crossover_rate: вероятность кроссовера
        elite_size: количество элитных особей
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bounds = np.array(bounds)
    n_dims = len(bounds)
    
    # Инициализация популяции
    population = np.random.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(population_size, n_dims)
    )
    
    def evaluate(pop):
        return np.array([f(ind) for ind in pop])
    
    def tournament_selection(pop, fitness, tournament_size=5):
        indices = np.random.choice(len(pop), tournament_size, replace=False)
        best_idx = indices[np.argmin(fitness[indices])]
        return pop[best_idx].copy()
    
    def crossover(parent1, parent2):
        if np.random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Арифметический кроссовер
        alpha = np.random.uniform(-0.5, 1.5)
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2
    
    def mutate(individual):
        for i in range(n_dims):
            if np.random.random() < mutation_rate:
                # Гауссова мутация
                mutation_range = (bounds[i, 1] - bounds[i, 0]) * 0.1
                individual[i] += np.random.normal(0, mutation_range)
                # Проверка границ
                individual[i] = np.clip(individual[i], bounds[i, 0], bounds[i, 1])
        return individual
    
    history = []
    best_overall = None
    best_fitness_overall = float('inf')
    
    for generation in range(n_generations):
        # Оценка
        fitness = evaluate(population)
        
        # Обновление лучшего решения
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness_overall:
            best_fitness_overall = fitness[best_idx]
            best_overall = population[best_idx].copy()
        
        history.append(best_fitness_overall)
        
        # Создание новой популяции
        new_population = []
        
        # Элитизм
        elite_indices = np.argsort(fitness)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
        
        # Заполнение остальной популяции
        while len(new_population) < population_size:
            # Селекция
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            
            # Кроссовер
            child1, child2 = crossover(parent1, parent2)
            
            # Мутация
            child1 = mutate(child1)
            child2 = mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = np.array(new_population[:population_size])
    
    return best_overall, best_fitness_overall, history


# ========== Метод роя частиц ==========

def particle_swarm_optimization(f, bounds, n_particles=30, max_iter=100,
                                 w=0.7, c1=1.5, c2=1.5, random_state=None):
    """
    Метод роя частиц (Particle Swarm Optimization, PSO).
    
    Алгоритм:
    1. Инициализация частиц со случайными позициями и скоростями
    2. Обновление личных и глобальных лучших позиций
    3. Обновление скоростей и позиций
    4. Повторять
    
    Параметры:
        w: коэффициент инерции
        c1: когнитивный коэффициент (к личной лучшей позиции)
        c2: социальный коэффициент (к глобальной лучшей позиции)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    bounds = np.array(bounds)
    n_dims = len(bounds)
    
    # Инициализация
    positions = np.random.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(n_particles, n_dims)
    )
    velocities = np.random.uniform(-1, 1, size=(n_particles, n_dims))
    
    # Личные лучшие позиции
    personal_best = positions.copy()
    personal_best_fitness = np.array([f(p) for p in positions])
    
    # Глобальная лучшая позиция
    global_best_idx = np.argmin(personal_best_fitness)
    global_best = personal_best[global_best_idx].copy()
    global_best_fitness = personal_best_fitness[global_best_idx]
    
    history = []
    
    for iteration in range(max_iter):
        # Обновление частиц
        for i in range(n_particles):
            # Случайные коэффициенты
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            
            # Обновление скорости
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best[i] - positions[i]) +
                           c2 * r2 * (global_best - positions[i]))
            
            # Ограничение скорости
            max_vel = (bounds[:, 1] - bounds[:, 0]) * 0.1
            velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
            
            # Обновление позиции
            positions[i] = positions[i] + velocities[i]
            
            # Проверка границ
            positions[i] = np.clip(positions[i], bounds[:, 0], bounds[:, 1])
            
            # Оценка
            fitness = f(positions[i])
            
            # Обновление личного лучшего
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best[i] = positions[i].copy()
                
                # Обновление глобального лучшего
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = positions[i].copy()
        
        history.append(global_best_fitness)
        
        # Адаптивное изменение инерции
        w = 0.9 - 0.5 * (iteration / max_iter)
    
    return global_best, global_best_fitness, history


def run_lab6():
    """Основная функция лабораторной работы."""
    print("=" * 60)
    print("Лабораторная работа №6: Методы глобальной оптимизации")
    print("=" * 60)
    
    # Параметры поиска
    bounds_sphere = [(-5, 5), (-5, 5)]
    bounds_rastrigin = [(-5.12, 5.12), (-5.12, 5.12)]
    bounds_ackley = [(-32, 32), (-32, 32)]
    
    x0 = np.array([2.5, 2.5])
    
    # Методы для сравнения
    methods = {
        'Random Search': lambda f, b: random_search(f, b, n_iterations=200, 
                                                     n_samples=20, random_state=42),
        'Simulated Annealing': lambda f, b: simulated_annealing(f, x0, b, 
                                                                  T0=1000, alpha=0.995,
                                                                  max_iter=500, random_state=42),
        'Genetic Algorithm': lambda f, b: genetic_algorithm(f, b, population_size=30,
                                                             n_generations=100, random_state=42),
        'PSO': lambda f, b: particle_swarm_optimization(f, b, n_particles=20,
                                                         max_iter=100, random_state=42),
    }
    
    all_results = {}
    
    # ========== Тест 1: Функция сферы ==========
    print("\n--- Тест 1: Функция сферы ---")
    print(f"Глобальный минимум: f(0, 0) = 0")
    
    histories_sphere = []
    labels = list(methods.keys())
    
    for name, method in methods.items():
        x_opt, f_opt, history = method(sphere, bounds_sphere)
        print(f"{name:20s}: f(x) = {f_opt:.6e}, x = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
        histories_sphere.append(history)
    
    all_results['sphere'] = (histories_sphere, labels)
    
    # ========== Тест 2: Функция Растригина ==========
    print("\n--- Тест 2: Функция Растригина ---")
    print(f"Глобальный минимум: f(0, 0) = 0")
    
    histories_rastrigin = []
    
    for name, method in methods.items():
        x_opt, f_opt, history = method(rastrigin, bounds_rastrigin)
        print(f"{name:20s}: f(x) = {f_opt:.6e}, x = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
        histories_rastrigin.append(history)
    
    all_results['rastrigin'] = (histories_rastrigin, labels)
    
    # ========== Тест 3: Функция Акли ==========
    print("\n--- Тест 3: Функция Акли ---")
    print(f"Глобальный минимум: f(0, 0) = 0")
    
    histories_ackley = []
    
    for name, method in methods.items():
        x_opt, f_opt, history = method(ackley, bounds_ackley)
        print(f"{name:20s}: f(x) = {f_opt:.6e}, x = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
        histories_ackley.append(history)
    
    all_results['ackley'] = (histories_ackley, labels)
    
    # ========== Визуализация ==========
    print("\n--- Визуализация результатов ---")
    
    fig1, _ = plot_comparison(
        histories_sphere, labels,
        'Методы глобальной оптимизации - Функция сферы'
    )
    save_plot(fig1, 'lab6_sphere_global.png', folder='lab_06/output')
    print("Сохранён график: lab_06/output/lab6_sphere_global.png")
    
    fig2, _ = plot_comparison(
        histories_rastrigin, labels,
        'Методы глобальной оптимизации - Функция Растригина'
    )
    save_plot(fig2, 'lab6_rastrigin_global.png', folder='lab_06/output')
    print("Сохранён график: lab_06/output/lab6_rastrigin_global.png")
    
    fig3, _ = plot_comparison(
        histories_ackley, labels,
        'Методы глобальной оптимизации - Функция Акли'
    )
    save_plot(fig3, 'lab6_ackley_global.png', folder='lab_06/output')
    print("Сохранён график: lab_06/output/lab6_ackley_global.png")
    
    print("\n" + "=" * 60)
    print("Лабораторная работа №6 завершена!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    run_lab6()
