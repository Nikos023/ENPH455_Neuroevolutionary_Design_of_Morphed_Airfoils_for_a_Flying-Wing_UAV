#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import os

# ============================================================
# === FUNCTIONS ==============================================
# ============================================================

def thickness_distribution(x, t):
    return 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

def compute_airfoil(x, yc, yt):
    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl

def smooth_camber(x_ctrl, y_ctrl, x_dense):
    spline = make_interp_spline(x_ctrl, y_ctrl, k=3)
    yc_dense = spline(x_dense)
    yc_dense = gaussian_filter1d(yc_dense, sigma=1.2)
    return yc_dense

def yc_base_function(x, m, p):
    return np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )

# ============================================================
# === BASE PARAMETERS ========================================
# ============================================================

m, p, t = 0.02, 0.4, 0.12
num_points = 1000
num_ctrl = 10

# Cosine spacing
beta = np.linspace(0, np.pi, num_points)
x = (1 - np.cos(beta)) / 2

# Base camber + thickness
yc_base = yc_base_function(x, m, p)
yt_base = thickness_distribution(x, t)

# Control points
n_each_side = num_ctrl // 2
x_ctrl_left = np.linspace(0, 1/3, n_each_side, endpoint=False)
x_ctrl_right = np.linspace(2/3, 1, n_each_side)
x_ctrl = np.concatenate([x_ctrl_left, x_ctrl_right])
y_ctrl_base = np.interp(x_ctrl, x, yc_base)

# NEAT-consistent offset limits
max_offsets = np.array([0.12,0.10,0.08,0.06,0.02,0.01,0.04,0.06,0.08,0.10])

# ============================================================
# === RANDOM SAMPLE GENERATION ===============================
# ============================================================

os.makedirs("Geometry", exist_ok=True)
n_samples = 5000

for i in range(1, n_samples + 1):

    # ---- Generate NEAT-style delta_y ----
    raw = np.random.normal(0, 0.5, size=num_ctrl)
    delta_y = raw * max_offsets
    delta_y = gaussian_filter1d(delta_y, sigma=2.0)

    # Apply to control points
    y_ctrl = y_ctrl_base + delta_y

    # ---- Smooth camber ----
    yc = smooth_camber(x_ctrl, y_ctrl, x)

    # ================== CENTER LOCK ==================
    center_start, center_end = 0.33, 0.66
    center_mask = (x > center_start) & (x < center_end)

    coeffs = np.polyfit(
        x[center_mask],
        yc_base_function(x, m, p)[center_mask],
        1
    )
    yc_trend = np.polyval(coeffs, x[center_mask])

    blend_x = (x[center_mask] - center_start) / (center_end - center_start)
    weights = 0.5 * (1 - np.cos(np.pi * blend_x))
    weights *= 0.6

    yc[center_mask] = (
        (1 - weights) * yc[center_mask] +
        weights * yc_trend
    )

    # Final smoothing
    yc = gaussian_filter1d(yc, sigma=2.0)

    # ---- Compute surfaces ----
    xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)

    # ========================================================
    # === SAVE FILES (UNCHANGED STRUCTURE) ====================
    # ========================================================

    base_name = f"airfoil_points_{i:04d}"
    txt_filename = f"Geometry/{base_name}.txt"
    dat_filename = f"Geometry/{base_name}.dat"

    # ---- TXT (same format as your original) ----
    with open(txt_filename, "w") as f:
        f.write("=== Airfoil Parameters ===\n")
        f.write(f"m = {m}\n")
        f.write(f"p = {p}\n")
        f.write(f"t = {t}\n\n")

        f.write("=== Control Points ===\n")
        for xi, yi, off in zip(x_ctrl, y_ctrl, delta_y):
            f.write(f"{xi:.5f}, {yi:.5f}, {off:.5f}\n")

        f.write("\n=== Upper Surface ===\n")
        for xi, yi in zip(xu, yu):
            f.write(f"{xi:.5f}, {yi:.5f}\n")

        f.write("\n=== Lower Surface ===\n")
        for xi, yi in zip(xl, yl):
            f.write(f"{xi:.5f}, {yi:.5f}\n")

    # ---- DAT (XFOIL) ----
    N = 100
    beta = np.linspace(0, np.pi, N)
    x_cos = 0.5 * (1 - np.cos(beta))

    y_upper_interp = np.interp(x_cos, xu, yu)
    y_lower_interp = np.interp(x_cos, xl, yl)

    x_all = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_all = np.concatenate([y_upper_interp[::-1], y_lower_interp[1:]])

    with open(dat_filename, "w") as f:
        f.write(f"{base_name}\n")
        for xi, yi in zip(x_all, y_all):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    print(f"✅ Saved {txt_filename} and {dat_filename}")

print("\n🎉 Done! Training-consistent airfoils generated in /Geometry/")