#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEAT Airfoil Sweep from Best Genome
- Reconstructs NEAT airfoil from genome for all AoA folders
- Matches training-consistent geometry processing
- Computes NeuralFoil CL/CD/CM
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import neat
from neuralfoil import get_aero_from_coordinates
import joblib

# ================================
# USER SETTINGS
# ================================
base_dir = "BestGenomes"
REYNOLDS = 1e6
re_folder = f"{REYNOLDS:.0e}".replace("+0","").replace("+","")
config_path = "NEAT Config Single Genome.ini"
num_ctrl = 10
num_points = 1000
beta = np.linspace(0, np.pi, num_points)
x_dense = (1 - np.cos(beta))/2

# NACA-like base parameters
m, p, t = 0.02, 0.4, 0.12
yt_base = 5 * t * (0.2969*np.sqrt(x_dense) - 0.126*x_dense - 0.3516*x_dense**2 + 0.2843*x_dense**3 - 0.1015*x_dense**4)

# Max delta_y per control point
max_offsets = np.array([0.12,0.10,0.08,0.02,0.001,0.001,0.02,0.08,0.10,0.12])
max_offsets = max_offsets * 0.75
n_each_side = num_ctrl // 2
x_ctrl = np.concatenate([np.linspace(0,1/3,n_each_side,endpoint=False), np.linspace(2/3,1,n_each_side)])

# ================================
# LOAD GLOBAL MODELS
# ================================
model_dir = os.path.join("../Comparison/Comparison Results/global_model/2000gb", f"Re{re_folder}")
model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb.joblib"))

# ================================
# HELPER FUNCTIONS
# ================================
def yc_base_function(x):
    return np.where(
        x < p,
        m/p**2*(2*p*x - x**2),
        m/(1-p)**2*((1-2*p)+2*p*x - x**2)
    )

def y_ctrl_base_function():
    return np.interp(x_ctrl, x_dense, yc_base_function(x_dense))

def smooth_camber(x_ctrl, y_ctrl, x_dense):
    spline = make_interp_spline(x_ctrl, y_ctrl, k=3)
    return gaussian_filter1d(spline(x_dense), sigma=1.2)

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

    X_input_gb = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2, AoA]).reshape(1,-1)
    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

    aero = get_aero_from_coordinates(
        coordinates=coords,
        alpha=[AoA],
        Re=REYNOLDS,
        model_size="xxxlarge",
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0
    )

    cl_nf, cd_nf, cm_nf = aero["CL"][0], aero["CD"][0], aero["CM"][0]

    if (AoA >= 0):
        cl_corr = cl_nf
        cd_corr = max(cd_nf, 1e-4)
        cm_corr = cm_nf

    else:
        cl_corr = cl_nf - model_cl.predict(X_input_gb)[0]
        cd_corr = max(cd_nf - model_cd.predict(X_input_gb)[0], 1e-3)
        cm_corr = cm_nf - model_cm.predict(X_input_gb)[0]

    return cl_corr, cd_corr, cm_corr

# ================================
# LOAD CONFIG & GENOME
# ================================
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# ================================
# FIND AoA FOLDERS
# ================================
re_dir = os.path.join(base_dir, f"Re{re_folder}")
aoa_folders = [f for f in os.listdir(re_dir) if f.endswith("Degrees") and os.path.isdir(os.path.join(re_dir,f))]
aoa_folders.sort(key=lambda s: float(s.split()[0]))

# ================================
# SWEEP
# ================================
plt.figure(figsize=(14,8))
colors = plt.cm.viridis(np.linspace(0,1,len(aoa_folders)))
found_any = False

aoa_vals, cl_vals, cd_vals, cm_vals, ld_vals = [], [], [], [], []

for color, aoa_folder in zip(colors, aoa_folders):

    best_genome_file = os.path.join(re_dir, aoa_folder, "best_genome_nf.pkl")
    if not os.path.exists(best_genome_file):
        print(f"Skipping {aoa_folder} (no genome found)")
        continue

    AoA = float(aoa_folder.split()[0])

    with open(best_genome_file, "rb") as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    y_ctrl_base = y_ctrl_base_function()
    X_input = np.hstack([y_ctrl_base, AoA]).reshape(1,-1)
    raw_output = np.array(net.activate(X_input.flatten()))[:num_ctrl]

    delta_y = np.clip(raw_output*max_offsets*2.0, -max_offsets, max_offsets)
    delta_y = gaussian_filter1d(delta_y, sigma=2.0)
    y_ctrl_new = y_ctrl_base + delta_y
    yc_new = smooth_camber(x_ctrl, y_ctrl_new, x_dense)

    # center lock (training-consistent)
    center_mask = (x_dense>0.33)&(x_dense<0.55)
    coeffs = np.polyfit(x_dense[center_mask], yc_base_function(x_dense)[center_mask], 1)
    yc_trend = np.polyval(coeffs, x_dense[center_mask])
    blend_x = (x_dense[center_mask]-0.33)/(0.55-0.45)
    weights = 0.5*(1-np.cos(np.pi*blend_x))*0.6
    yc_new[center_mask] = (1-weights)*yc_new[center_mask]+weights*yc_trend
    yc_new = gaussian_filter1d(yc_new, sigma=2.0)

    xu, yu, xl, yl = compute_airfoil(x_dense, yc_new, yt_base)

    # Plot airfoil surfaces
    plt.plot(xu, yu, color=color, label=aoa_folder)
    plt.plot(xl, yl, color=color)

    # Plot camber line (black dashed, no legend)
    plt.plot(x_dense, yc_new, color='black', linestyle='--', linewidth=0.5, alpha=1)
    found_any = True

    cl, cd, cm = apply_gb_correction(delta_y, xu, yu, xl, yl, AoA)
    aoa_vals.append(AoA)
    cl_vals.append(cl)
    cd_vals.append(cd)
    cm_vals.append(cm)
    ld_vals.append(cl/cd)

    print(f"AoA {AoA:.2f}° → CL={cl:.6f} CD={cd:.6f} CM={cm:.6f} CL/CD={cl/cd:.2f}")

if not found_any:
    raise RuntimeError("No genomes found.")

plt.axis("equal")
plt.grid(True)
plt.xlabel("x/Chord (c)",  fontweight='bold')
plt.ylabel("y/Chord (c)",  fontweight='bold')
plt.title(f"NEAT Airfoils Overlay @ Re={re_folder}", fontsize=16, fontweight='bold')

norm = mpl.colors.Normalize(vmin=min(aoa_vals), vmax=max(aoa_vals))
sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

cbar = plt.colorbar(
    sm,
    ax=plt.gca(),
    orientation="horizontal",
    pad=0.08
)

cbar.set_label("Angle of Attack (°)", fontweight='bold')

# ================================
# LOAD XFOIL DATA
# ================================
import pandas as pd

xfoil_file = os.path.join("XFOIL Results", "NACA2412Re1e6.txt")

# Skip header lines and load table
with open(xfoil_file, "r") as f:
    lines = f.readlines()

# Find the start of the data table
for i, line in enumerate(lines):
    if line.strip().startswith("Alpha"):
        start_idx = i + 1
        break

data_xfoil = pd.read_csv(
    xfoil_file,
    skiprows=start_idx,
    names=["Alpha","Cl","Cd","Cdp","Cm","Top_Xtr","Bot_Xtr"]
)

# Filter to AoA between -5 and 12.5 degrees
xfoil_filtered = data_xfoil[(data_xfoil["Alpha"] >= -5) & (data_xfoil["Alpha"] <= 12.5)]

# Compute CL/CD safely without SettingWithCopyWarning
xfoil_filtered = xfoil_filtered.copy()  # make a true copy first
xfoil_filtered["CL_CD"] = xfoil_filtered["Cl"] / xfoil_filtered["Cd"]

# ================================
# PLOT CL/CD with XFOIL overlay
# ================================
plt.figure(figsize=(10,6))
plt.plot(aoa_vals, ld_vals, 'o-', label="NEAT Airfoil", color="tab:blue")
plt.plot(
    xfoil_filtered["Alpha"],
    xfoil_filtered["CL_CD"],
    's--', lw=2, label="NACA2412 XFOIL", color="tab:orange"
)
plt.grid(True)
plt.xlabel("Angle of Attack (°)",  fontweight='bold')
plt.ylabel("CL/CD",  fontweight='bold')
plt.title(f"CL/CD vs AoA @ Re={re_folder}", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# PLOT CM with XFOIL overlay
# ================================
plt.figure(figsize=(10,6))
plt.plot(aoa_vals, cm_vals, 'o-', label="NEAT Airfoil", color="tab:blue")
plt.plot(
    xfoil_filtered["Alpha"],
    xfoil_filtered["Cm"],
    's--', lw=2, label="NACA2412 XFOIL", color="tab:orange"
)
plt.grid(True)
plt.xlabel("Angle of Attack (°)",  fontweight='bold')
plt.ylabel("CM",  fontweight='bold')
plt.title(f"Pitching Moment vs AoA @ Re={re_folder}", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()