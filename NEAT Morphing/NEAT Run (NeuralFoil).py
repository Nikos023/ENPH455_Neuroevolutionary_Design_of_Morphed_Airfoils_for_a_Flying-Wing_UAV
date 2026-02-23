#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEAT Airfoil Tester with NeuralFoil + GB Correction (Training-Consistent Geometry(Not))
- Matches geometry processing used during NEAT training
- Ensures CM, CL, CD consistency with convergence plots
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
AoA = 5.0

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
yt_base = 5 * t * (
    0.2969*np.sqrt(x_dense)
    - 0.126*x_dense
    - 0.3516*x_dense**2
    + 0.2843*x_dense**3
    - 0.1015*x_dense**4
)

#max_offsets = np.array([0.05,0.04,0.03,0.02,0.01,0.01,0.02,0.03,0.04,0.05])
max_offsets = np.array([0.12,0.10,0.08,0.06,0.2,0.01,0.04,0.06,0.08,0.10])
#max_offsets = np.array([0.15,0.12,0.09,0.06,0.01,0.01,0.06,0.09,0.12,0.15])
#max_offsets = np.array([0.20,0.16,0.12,0.09,0.01,0.01,0.09,0.012,0.16,0.20])

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
    yc = spline(x_dense)
    return gaussian_filter1d(yc, sigma=1.2)

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
    return np.vstack([coords_upper, coords_lower[1:]])

def apply_gb_correction(delta_y, xu, yu, xl, yl, AoA):
    dy_vec = delta_y
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)

    X_input_gb = np.hstack(
        [dy_vec, dy_cumsum, dy_dx, d2y_dx2, AoA]
    ).reshape(1, -1)

    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

    aero = get_aero_from_coordinates(
        coordinates=coords,
        alpha=[AoA],
        Re=5e5,
        model_size="xxxlarge",
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0
    )

    cl_nf = aero["CL"][0]
    cd_nf = aero["CD"][0]
    cm_nf = aero["CM"][0]

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

if not os.path.exists(best_genome_file):
    raise FileNotFoundError("No trained genome found.")

with open(best_genome_file, "rb") as f:
    genome = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(genome, config)

# ================== PREDICT OFFSETS =====================
y_ctrl_base = y_ctrl_base_function()
X_input = np.hstack([y_ctrl_base, AoA]).reshape(1, -1)
raw_output = np.array(net.activate(X_input.flatten()))[:num_ctrl]

delta_y = np.clip(raw_output * max_offsets * 2.0, -max_offsets, max_offsets)
delta_y = gaussian_filter1d(delta_y, sigma=2.0)  # MATCH TRAINING

y_ctrl_new = y_ctrl_base + delta_y
yc_new = smooth_camber(x_ctrl, y_ctrl_new, x_dense)

# ================== TRAINING-CONSISTENT CENTER LOCK ==================
center_start, center_end = 0.33, 0.66
center_mask = (x_dense > center_start) & (x_dense < center_end)

coeffs = np.polyfit(
    x_dense[center_mask],
    yc_base_function(x_dense)[center_mask],
    1
)
yc_trend = np.polyval(coeffs, x_dense[center_mask])

blend_x = (x_dense[center_mask] - center_start) / (center_end - center_start)
weights = 0.5 * (1 - np.cos(np.pi * blend_x))
weights *= 0.6  # MATCH TRAINING

yc_new[center_mask] = (
    (1 - weights) * yc_new[center_mask] +
    weights * yc_trend
)

yc_new = gaussian_filter1d(yc_new, sigma=2.0)  # MATCH TRAINING

xu, yu, xl, yl = compute_airfoil(x_dense, yc_new, yt_base)

# ================== PLOTS ==================
plt.figure(figsize=(12, 6))
plt.plot(xu, yu, 'b-', lw=2, label="Upper Surface")
plt.plot(xl, yl, 'b-', lw=2, label="Lower Surface")
plt.plot(x_dense, yc_new, 'r--', lw=1.5, label="Camber Line")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x (chord)")
plt.ylabel("y")
plt.title(f"NEAT Airfoil (Training-Consistent) @ AoA = {AoA}°")
plt.legend()
plt.show()

# ================== ROTATED AIRFOIL @ AoA ==================
theta = -np.deg2rad(AoA)

# Rotation matrix about origin (leading edge)
def rotate(x, y, theta):
    x_rot = x*np.cos(theta) - y*np.sin(theta)
    y_rot = x*np.sin(theta) + y*np.cos(theta)
    return x_rot, y_rot

xu_rot, yu_rot = rotate(xu, yu, theta)
xl_rot, yl_rot = rotate(xl, yl, theta)
xc_rot, yc_rot = rotate(x_dense, yc_new, theta)

plt.figure(figsize=(12, 6))
plt.plot(xu_rot, yu_rot, 'b-', lw=2, label="Upper Surface (Rotated)")
plt.plot(xl_rot, yl_rot, 'b-', lw=2, label="Lower Surface (Rotated)")
plt.plot(xc_rot, yc_rot, 'r--', lw=1.5, label="Camber Line (Rotated)")
plt.axhline(0, linestyle='--', linewidth=1)
plt.axis("equal")
plt.grid(True)
plt.xlabel("x (global)")
plt.ylabel("y (global)")
plt.title(f"NEAT Airfoil Physically Rotated to AoA = {AoA}°")
plt.legend()
plt.show()

# ================== NACA 2412 CAMBER LINE ==================
yc_naca = yc_base_function(x_dense)

# ================== OVERLAY WITH NACA 2412 ==================
naca_file = "../Morphing/airfoil_xfoil_2412.dat"
data = np.loadtxt(naca_file, skiprows=1)
x_ref, y_ref = data[:, 0], data[:, 1]

plt.figure(figsize=(12, 6))
plt.plot(xu, yu, 'b-', lw=2, label="Upper Surface (NEAT)")
plt.plot(xl, yl, 'b-', lw=2, label="Lower Surface (NEAT)")
plt.plot(x_dense, yc_new, 'r--', lw=1.5, label="Camber Line (NEAT)")
plt.plot(x_dense, yc_naca, 'g-.', lw=1.5, label="Camber Line (NACA 2412)")
plt.plot(x_ref, y_ref, 'k--', lw=1.5, label="NACA 2412 (surface)")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x (chord)")
plt.ylabel("y")
plt.title(f"NEAT Airfoil vs NACA 2412 @ AoA = {AoA}°")
plt.legend()
plt.show()

# ================== SAVE FILES ==========================
os.makedirs("Geometry", exist_ok=True)

txt_filename = f"Geometry/{output_name}.txt"
with open(txt_filename, "w") as f:
    f.write("=== Control Points + Offsets ===\n")
    for xi, yi, off in zip(x_ctrl, y_ctrl_new, delta_y):
        f.write(f"{xi:.5f}, {yi:.5f}, {off:.5f}\n")

dat_filename = f"Geometry/{output_name}.dat"
N = 100
beta_cos = np.linspace(0, np.pi, N)
x_cos = 0.5*(1 - np.cos(beta_cos))
y_upper = np.interp(x_cos, xu, yu)
y_lower = np.interp(x_cos, xl, yl)
x_all = np.concatenate([x_cos[::-1], x_cos[1:]])
y_all = np.concatenate([y_upper[::-1], y_lower[1:]])

with open(dat_filename, "w") as f:
    f.write(f"{output_name}\n")
    for xi, yi in zip(x_all, y_all):
        f.write(f"{xi:.6f} {yi:.6f}\n")

# ================== AERODYNAMICS ==================
cl_corr, cd_corr, cm_corr = apply_gb_correction(delta_y, xu, yu, xl, yl, AoA)
print(f"📊 Corrected @ AoA={AoA}° → CL={cl_corr:.6f}, CD={cd_corr:.6f}, CM={cm_corr:.10f}, CL/CD={cl_corr/cd_corr:.6f}")