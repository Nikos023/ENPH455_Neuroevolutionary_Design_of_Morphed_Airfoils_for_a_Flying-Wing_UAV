#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEAT Airfoil Tester with NeuralFoil + GB Correction (Smooth Geometry)
- Loads a trained genome (best_genome_nf_aoaX.pkl)
- Predicts control point offsets
- Generates smoothly morphed airfoil geometry
- Applies smooth center-region lock + centerline lock
- Evaluates corrected aerodynamic coefficients using NeuralFoil + GB models
- Plots smooth airfoil and saves .txt and .dat files for XFOIL
"""

import neat
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import os
import joblib
from neuralfoil import get_aero_from_coordinates

# ================== CONFIG ===============================
config_path = "NEAT Config Single Genome.ini"
best_genome_file = "BestGenomes/best_genome_nf_aoa5.pkl"
output_name = "NEAT_airfoil_10ctrl"
AoA = 5.0  # target angle of attack in degrees

num_ctrl = 10
n_each_side = num_ctrl // 2
x_ctrl_left = np.linspace(0, 1/3, n_each_side, endpoint=False)
x_ctrl_right = np.linspace(2/3, 1, n_each_side)
x_ctrl = np.concatenate([x_ctrl_left, x_ctrl_right])

num_points = 1000
beta = np.linspace(0, np.pi, num_points)
x_dense = (1 - np.cos(beta)) / 2

# Base airfoil parameters
m, p, t = 0.02, 0.4, 0.12
yt_base = 5 * t * (0.2969*np.sqrt(x_dense) - 0.126*x_dense
                   - 0.3516*x_dense**2 + 0.2843*x_dense**3
                   - 0.1015*x_dense**4)

# Max delta_y for each control point (as in training)
max_offsets = np.array([0.05,0.04,0.03,0.02,0.01,0.01,0.02,0.03,0.04,0.05])

# ================== LOAD GB MODELS ========================
model_dir = "../Comparison/Comparison Results/global_model"
model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb_2000_samples.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb_2000_samples.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb_2000_samples.joblib"))

# ================== HELPER FUNCTIONS ====================
def yc_base_function(x):
    return np.where(
        x < p,
        m / p**2 * (2*p*x - x**2),
        m / (1 - p)**2 * ((1 - 2*p) + 2*p*x - x**2)
    )

def y_ctrl_base_function():
    yc_base = yc_base_function(x_dense)
    return np.interp(x_ctrl, x_dense, yc_base)

def smooth_camber(x_ctrl, y_ctrl, x_dense):
    spline = make_interp_spline(x_ctrl, y_ctrl, k=3)
    yc_dense = spline(x_dense)
    return gaussian_filter1d(yc_dense, sigma=1.2)

def compute_airfoil(x, yc, yt):
    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)
    xu = x - yt*np.sin(theta)
    yu = yc + yt*np.cos(theta)
    xl = x + yt*np.sin(theta)
    yl = yc - yt*np.cos(theta)
    return xu, yu, xl, yl

def prepare_coordinates_for_neuralfoil(xu, yu, xl, yl):
    coords_upper = np.vstack([xu[::-1], yu[::-1]]).T
    coords_lower = np.vstack([xl, yl]).T
    coords = np.vstack([coords_upper, coords_lower[1:]])
    return coords

def apply_gb_correction(delta_y, xu, yu, xl, yl, AoA):
    dy_vec = delta_y
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)
    X_input_gb = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2, AoA]).reshape(1, -1)

    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

    try:
        aero_nf = get_aero_from_coordinates(
            coordinates=coords,
            alpha=[AoA],
            Re=1e6,
            model_size="xxxlarge",
            n_crit=9.0,
            xtr_upper=1.0,
            xtr_lower=1.0
        )
        cl_nf = aero_nf["CL"][0]
        cd_nf = aero_nf["CD"][0]
        cm_nf = aero_nf["CM"][0]
    except Exception as e:
        print(f"⚠️ NeuralFoil evaluation failed: {e}")
        return None, None, None

    cl_corr = cl_nf - model_cl.predict(X_input_gb)[0]
    cd_corr = max(cd_nf - model_cd.predict(X_input_gb)[0], 1e-5)
    cm_corr = cm_nf - model_cm.predict(X_input_gb)[0]

    return cl_corr, cd_corr, cm_corr

# ================== LOAD CONFIG & GENOME =================
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

if best_genome_file and os.path.exists(best_genome_file):
    print(f"📂 Loading trained genome from {best_genome_file}...")
    with open(best_genome_file, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # ================== PREDICT OFFSETS =====================
    y_ctrl_base = y_ctrl_base_function()
    X_input = np.hstack([y_ctrl_base, AoA]).reshape(1, -1)
    raw_output = np.array(net.activate(X_input.flatten()))[:num_ctrl]
    delta_y = np.clip(raw_output * max_offsets * 2.0, -max_offsets, max_offsets)

    # Apply offsets and smooth
    y_ctrl_new = y_ctrl_base + delta_y
    yc_new = smooth_camber(x_ctrl, y_ctrl_new, x_dense)

    # ================== SMOOTH CENTER LOCK (EVEN SMOOTHER) ==================
    center_start, center_end = 0.3, 0.7  # wider blending region
    center_mask = (x_dense > center_start) & (x_dense < center_end)
    blend_x = (x_dense[center_mask] - center_start) / (center_end - center_start)
    weights = 0.5 * (1 - np.cos(np.pi * blend_x))  # cosine taper
    yc_new[center_mask] = (1 - weights) * yc_new[center_mask] + weights * yc_base_function(x_dense)[center_mask]

    # ================== SMOOTH CAMBER LINE ==================
    yc_new = gaussian_filter1d(yc_new, sigma=8.0)  # increased sigma for smoother camber

    # ================== CENTERLINE LOCK =====================
    mid_mask = (x_dense > 0.4) & (x_dense < 0.6)
    yc_new -= np.mean(yc_new[mid_mask])

    xu, yu, xl, yl = compute_airfoil(x_dense, yc_new, yt_base)

    # ================== PLOT SMOOTH AIRFOIL ==================
    plt.figure(figsize=(12, 6))
    plt.plot(xu, yu, 'b-', lw=2, label="Upper Surface")
    plt.plot(xl, yl, 'b-', lw=2, label="Lower Surface")
    plt.plot(x_dense, yc_new, 'r--', lw=1.5, label="Camber Line (smoothed)")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x (chord)")
    plt.ylabel("y")
    plt.title(f"NEAT Predicted Airfoil - AoA = {AoA}°")
    plt.legend()
    plt.show()

    # ================== SAVE FILES ==========================
    os.makedirs("Geometry", exist_ok=True)

    # TXT
    txt_filename = f"Geometry/{output_name}.txt"
    with open(txt_filename, "w") as f:
        f.write("=== Airfoil Parameters ===\n")
        f.write(f"m = {m}\n")
        f.write(f"p = {p}\n")
        f.write(f"t = {t}\n\n")
        f.write("=== Control Points + Offsets ===\n")
        for xi, yi, off in zip(x_ctrl, y_ctrl_new, delta_y):
            f.write(f"{xi:.5f}, {yi:.5f}, {off:.5f}\n")
        f.write("\n=== Upper Surface ===\n")
        for xi, yi in zip(xu, yu):
            f.write(f"{xi:.5f}, {yi:.5f}\n")
        f.write("\n=== Lower Surface ===\n")
        for xi, yi in zip(xl, yl):
            f.write(f"{xi:.5f}, {yi:.5f}\n")
    print(f"✅ Airfoil saved as {txt_filename}")

    # DAT for XFOIL
    N = 100
    beta_cos = np.linspace(0, np.pi, N)
    x_cos = 0.5*(1 - np.cos(beta_cos))
    y_upper_interp = np.interp(x_cos, xu, yu)
    y_lower_interp = np.interp(x_cos, xl, yl)
    x_all = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_all = np.concatenate([y_upper_interp[::-1], y_lower_interp[1:]])
    dat_filename = f"Geometry/{output_name}.dat"
    with open(dat_filename, "w") as f:
        f.write(f"{output_name}\n")
        for xi, yi in zip(x_all, y_all):
            f.write(f"{xi:.6f} {yi:.6f}\n")
    print(f"✅ XFOIL-compatible file saved as {dat_filename}")

    # ================== NEURALFOIL + GB CORRECTION ==================
    cl_corr, cd_corr, cm_corr = apply_gb_correction(delta_y, xu, yu, xl, yl, AoA)
    if cl_corr is not None:
        print(f"📊 Corrected Aerodynamics @ AoA={AoA}° -> CL: {cl_corr:.4f}, CD: {cd_corr:.6f}, CM: {cm_corr:.4f}")
    else:
        print("⚠️ NeuralFoil + GB correction failed.")

else:
    print("⚠️ No trained genome found. Please run training first.")