#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:19:25 2026

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from dataclasses import dataclass

# -----------------------------------
# Константы
# -----------------------------------

mu0 = 4 * np.pi * 1e-7

M = 6.44e5

d = 0.020 # 20 мм

U = 10000 # напряжение

t0 = perf_counter()  #Время расчета

rho = 7500

# -----------------------------------
# Модель
# -----------------------------------

def B_field(L, k):
    D = d / k
    term1 = 1.0 / np.sqrt(D**2 + L**2)
    term2 = 1.0 / np.sqrt(d**2 + L**2)
    return np.abs(mu0 * M * L * (term1 - term2))

def alpha(L, k):
    B = B_field(L, k)
    T = 2 * (L + 0.5 * L )
    #t = 0.5 * L # зазор между магнитами (варьируется от 0,5L до L)
    return 2.8e8 * B**2 * T**2 / U

def magnet_mass(L, k):
    D = d / k
    V = np.pi / 4 * (D**2 - d**2) * L
    return rho * V

# ======================================
# Поиск фронта Парето через функцию
# ======================================

@dataclass
class ParetoZone:

    L: np.ndarray
    K: np.ndarray

    B: np.ndarray
    A: np.ndarray

    idx_B: int
    idx_A: int
    idx_K: int

def pareto_front(B, A):

    n = len(B)

    pareto = np.ones(n, dtype=bool)

    for i in range(n):

        if not pareto[i]:
            continue

        dominates = (
            (B >= B[i]) &
            (A <= A[i]) &
            (
                (B > B[i]) |
                (A < A[i])
            )
        )

        dominates[i] = False

        if np.any(dominates):
            pareto[i] = False

    return pareto

# ======================================
# Knee Point
# ======================================

def find_knee(A, B):
    if len(A) < 3:
        return 0

    order = np.lexsort((B, A))
    A_ord = A[order]
    B_ord = B[order]

    eps = np.finfo(float).eps
    A_norm = (A_ord - A_ord.min()) / (A_ord.max() - A_ord.min() + eps)
    B_norm = (B_ord - B_ord.min()) / (B_ord.max() - B_ord.min() + eps)

    p1 = np.array([A_norm[0], B_norm[0]])
    p2 = np.array([A_norm[-1], B_norm[-1]])

    v = p2 - p1
    norm_v = np.linalg.norm(v)

    if norm_v == 0:
        return 0

    dist = np.zeros(len(A_ord))

    for i in range(len(A_ord)):
        w = np.array([A_norm[i], B_norm[i]]) - p1
        dist[i] = abs(v[0]*w[1] - v[1]*w[0]) / norm_v

    return order[np.argmax(dist)]

# -----------------------------------
# Сетка
# -----------------------------------

N = 400

L_vals = np.linspace(0.001, 0.03, N)
k_vals = np.linspace(0.5, 0.95, N)

K, L = np.meshgrid(k_vals, L_vals)

B_grid = B_field(L, K)
A_grid = alpha(L, K)

Mass_grid = magnet_mass(L, K)

B_flat = B_grid.flatten()
A_flat = A_grid.flatten()
L_flat = L.flatten()
K_flat = K.flatten()

# Общий фронт Парето (для всех допустимых точек)
global_pareto_mask = pareto_front(B_flat, A_flat)
B_pareto = B_flat[global_pareto_mask]
A_pareto = A_flat[global_pareto_mask]
L_pareto = L_flat[global_pareto_mask]
K_pareto = K_flat[global_pareto_mask]

print(f"Pareto time = {perf_counter()-t0:.3f} s")

# ======================================
# Фильтр зоны пропускания
# ======================================

def optimize_zone(alpha_min=None, alpha_max=None):

    mask = np.ones_like(A_pareto, dtype=bool)

    if alpha_min is not None:
        mask &= (A_pareto >= alpha_min)

    if alpha_max is not None:
        mask &= (A_pareto <= alpha_max)

    L_p = L_pareto[mask]
    K_p = K_pareto[mask]
    B_p = B_pareto[mask]
    A_p = A_pareto[mask]

    if len(B_p) == 0:
        return None
    
    sort_idx = np.argsort(A_p)

    L_p = L_p[sort_idx]
    K_p = K_p[sort_idx]
    B_p = B_p[sort_idx]
    A_p = A_p[sort_idx]

    idx_B = np.argmax(B_p)
    idx_A = np.argmin(A_p)
    idx_K = find_knee(A_p, B_p)

    return ParetoZone(
        L=L_p,
        K=K_p,
        B=B_p,
        A=A_p,
        idx_B=idx_B,
        idx_A=idx_A,
        idx_K=idx_K
    )

zones = {
    "Зона 1": optimize_zone(alpha_max=0.66),
    "Зона 2": optimize_zone(alpha_min=1.72,
                            alpha_max=3.76)
        }

colors_points = {'Зона 1': ('red', 'green', 'black'), 'Зона 2': ('orange', 'lime', 'purple')}
colors_curves = {'Зона 1': 'cyan', 'Зона 2': 'magenta'}

# ======================================
# Для проверки массы
# ======================================

print("\n--- Масса магнита ---")
print(f"Минимальная масса: {Mass_grid.min() * 1000:.3f} г")
print(f"Максимальная масса: {Mass_grid.max() * 1000:.3f} г")

for k in [0.95, 0.8, 0.5, 0.3]:
    m = magnet_mass(0.02, k)

    print(
        f"L = 20 мм, k = {k:.2f}, "
        f"D = {d/k*1000:.2f} мм, "
        f"m = {m*1000:.3f} г"
    )

# ======================================
# Вывод точек
# ======================================


for name, z in zones.items():
    print(f"=== {name} ===")
    if z is None:
        print("Решений в данной зоне не найдено.\n")
        continue
    print(f"Maximum B:     L = {z.L[z.idx_B]*1000:.2f} мм, k = {z.K[z.idx_B]:.3f}")
    print(f"Minimum alpha: L = {z.L[z.idx_A]*1000:.2f} мм, k = {z.K[z.idx_A]:.3f}")
    print(f"Knee Point:    L = {z.L[z.idx_K]*1000:.2f} мм, k = {z.K[z.idx_K]:.3f}\n")

# ======================================
# Функции построения 3D графиков
# ======================================

def plot_surface(
        Z,
        zlabel,
        title,
        z_key):

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        K, L*1000, Z, cmap='viridis', alpha=0.4)

    Z_pareto = Z.flatten()[global_pareto_mask]
    #ax.scatter(K_pareto, L_pareto * 1000, Z_pareto, color="red", s=10)

    plot_zone_curves(
        ax,
        zones,
        colors_curves,
        x_key='K',
        y_key='L',
        z_key=z_key,
        scale_y=1000
    )

    plot_zone_key_points(
        ax,
        zones,
        colors_points,
        x_key='K',
        y_key='L',
        z_key=z_key,
        scale_y=1000
    )

    ax.set_box_aspect((1,1, 0.7), zoom=0.91)
    ax.view_init(25,-50)
    ax.dist = 15

    ax.set_xlabel("k")
    ax.set_ylabel("L, мм")
    ax.set_zlabel(zlabel)

    ax.set_title(title)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# ======================================
# Функции отрисовки Фронта Парето и Зон пропускания
# ======================================

def plot_zone_curves(ax, zones_dict, curve_colors, x_key='K', y_key='L', z_key=None, scale_y=1.0, scale_z=1.0):

    for name, z in zones_dict.items():
        if z is None:
            continue
        color = curve_colors[name]
        
        x = getattr(z, x_key)
        y = getattr(z, y_key) * scale_y
        
        if z_key is None:
            ax.plot(x, y, color=color, linewidth=2.5, label=f'Фронт Парето ({name})', zorder=4)
        else:
            z_vals = getattr(z, z_key) * scale_z
            ax.plot(x, y, z_vals, color=color, linewidth=3, label=f'Фронт Парето ({name})', zorder=4)

def plot_zone_key_points(ax, zones_dict, point_colors, x_key='K', y_key='L', z_key=None, scale_y=1.0, scale_z=1.0, add_labels=True):
    for name, z in zones_dict.items():
        if z is None:
            continue
            
        c_b, c_a, c_k = point_colors[name]
        
        x_arr = getattr(z, x_key)
        y_arr = getattr(z, y_key) * scale_y

        x_b, x_a, x_k = x_arr[z.idx_B], x_arr[z.idx_A], x_arr[z.idx_K]
        y_b, y_a, y_k = y_arr[z.idx_B], y_arr[z.idx_A], y_arr[z.idx_K]
        
        lbl_b = f'Max B ({name})' if add_labels else None
        lbl_a = f'Min α ({name})' if add_labels else None
        lbl_k = f'Knee Point ({name})' if add_labels else None

        if z_key is None:
            ax.scatter(x_b, y_b, s=180, color=c_b, marker='*', label=lbl_b, zorder=6)
            ax.scatter(x_a, y_a, s=160, color=c_a, marker='^', label=lbl_a, zorder=6)
            ax.scatter(x_k, y_k, s=160, color=c_k, marker='D', label=lbl_k, zorder=6)
        else:
            z_arr = getattr(z, z_key) * scale_z
            z_b, z_a, z_k = z_arr[z.idx_B], z_arr[z.idx_A], z_arr[z.idx_K]
            
            ax.scatter(x_b, y_b, z_b, s=180, color=c_b, marker='*', label=lbl_b, zorder=6)
            ax.scatter(x_a, y_a, z_a, s=160, color=c_a, marker='^', label=lbl_a, zorder=6)
            ax.scatter(x_k, y_k, z_k, s=160, color=c_k, marker='D', label=lbl_k, zorder=6)

# ======================================
# График 1: Критерии (A vs B)
# ======================================

plt.figure(figsize=(10, 6))
ax = plt.gca()

plt.scatter(A_flat, B_flat, s=8, color='lightgray', alpha=0.35, label='Все решения')
plt.scatter(A_pareto, B_pareto, s=25, color='royalblue', label='Общий фронт Парето')

plot_zone_curves(ax, zones, colors_curves, x_key='A', y_key='B')
plot_zone_key_points(ax, zones, colors_points, x_key='A', y_key='B', add_labels=True)

plt.xlabel(r'$\alpha$')
plt.ylabel('B, Тл')
plt.title('Пространство решений')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ======================================
# График 2: Параметры (k vs L)
# ======================================

plt.figure(figsize=(10, 6))
ax = plt.gca()

plt.scatter(K_flat, L_flat * 1000, s=6, color='lightgray', alpha=0.3, label='Все решения')
plt.scatter(K_pareto, L_pareto * 1000, s=20, color='royalblue', label='Фронт Парето')

plot_zone_curves(ax, zones, colors_curves, x_key='K', y_key='L', scale_y=1000)
plot_zone_key_points(ax, zones, colors_points, x_key='K', y_key='L', scale_y=1000, add_labels=True)

plt.xlabel('k = d/D')
plt.ylabel('L, мм')
plt.title('Пространство параметров')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ======================================
# График 3D: Поверхность B
# ======================================

plot_surface(B_grid, "B, Тл", "Поверхность B", "B")

# ======================================
# График 3D: Поверхность A
# ======================================

plot_surface(A_grid, r"$\alpha$", "Поверхность α", "A")

# ======================================
# График 3D: Поверхность m(L,k)
# ======================================

fig = plt.figure(figsize=(10, 7))

ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    K,
    L * 1000,
    Mass_grid * 1000,
    cmap="viridis",
    edgecolor="none"
)

ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$L$, мм")
ax.set_zlabel(r"$m$, г")
ax.set_title("Масса кольцевого магнита")

fig.colorbar(
    surf,
    ax=ax,
    shrink=0.7,
    label="Масса, г"
)

plt.tight_layout()
plt.show()

# ======================================
# График 3D: Поверхность m(L при фиксир значениях,k)
# ======================================

plt.figure(figsize=(9, 6))

for L_mm in [5, 10, 20, 30]:
    L = L_mm / 1000

    mass = magnet_mass(L, k_vals) * 1000

    plt.plot(
        k_vals,
        mass,
        label=f"L = {L_mm} мм"
    )

plt.xlabel(r"$k$")
plt.ylabel(r"$m$, г")
plt.title("Зависимость массы магнита от коэффициента k")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ======================================
# График 3D: Поверхность m(L)
# ======================================

plt.figure(figsize=(9, 6))

for k in [0.3, 0.5, 0.8, 0.95]:
    mass = magnet_mass(L_vals, k) * 1000

    plt.plot(
        L_vals * 1000,
        mass,
        label=fr"$k={k}$"
    )

plt.xlabel(r"$L$, мм")
plt.ylabel(r"$m$, г")
plt.title("Зависимость массы магнита от толщины L")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
