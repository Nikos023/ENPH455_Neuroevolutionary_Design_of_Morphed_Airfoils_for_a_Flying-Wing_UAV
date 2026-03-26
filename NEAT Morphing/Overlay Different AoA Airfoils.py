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
from sklearn.decomposition import PCA
import neat
from neuralfoil import get_aero_from_coordinates
import joblib

# ================================
# USER SETTINGS
# ================================
base_dir = "BestGenomes"
REYNOLDS = 1e5
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
max_offsets = np.array([0.12,0.10,0.08,0.04,0.01,0.01,0.02,0.08,0.10,0.12])
max_offsets = max_offsets * 0.65
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

yc_base = np.where(
    x_dense < p,
    m/p**2*(2*p*x_dense - x_dense**2),
    m/(1-p)**2*((1-2*p)+2*p*x_dense - x_dense**2)
)

def y_ctrl_base_function():
    return np.interp(x_ctrl, x_dense, yc_base)

def smooth_camber(x_ctrl, y_ctrl, x_dense):
    spline = make_interp_spline(x_ctrl, y_ctrl, k=1)
    return gaussian_filter1d(spline(x_dense), sigma=25)

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

    cl_corr = cl_nf
    cd_corr = max(cd_nf, 1e-3)
    cm_corr = cm_nf - model_cm.predict(X_input_gb)[0]

    return cl_corr, cd_corr, cm_corr

def parse_cp(file):
    data = np.loadtxt(file, skiprows=3)
    return data[:,0], data[:,1]

def parse_polar(file):
    """
    Reads XFOIL polar.dat with single AoA entry
    Returns: CL, CD, CM
    """
    with open(file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # Look for numeric data row (starts with alpha value)
        if len(parts) >= 5:
            try:
                alpha = float(parts[0])  # not used, but confirms valid row
                cl = float(parts[1])
                cd = float(parts[2])
                cm = float(parts[4])
                return cl, cd, cm
            except:
                continue

    raise ValueError(f"Could not parse polar file: {file}")

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

aoa_vals, cl_vals, cd_vals, cm_vals, ld_vals, cp_data, geom_data = [], [], [], [], [], [], []
polar_cl_vals, polar_cd_vals, polar_cm_vals, polar_ld_vals = [], [], [], []
delta_y_data = []

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

    delta_y = raw_output * max_offsets * 2.0
    delta_y = gaussian_filter1d(delta_y, sigma=2.0)
    delta_y = np.clip(delta_y, -max_offsets, max_offsets)

    delta_y_data.append((AoA, delta_y.copy()))

    y_ctrl_new = y_ctrl_base + delta_y
    yc_new = smooth_camber(x_ctrl, y_ctrl_new, x_dense)

    # center lock (training-consistent)
    center_start, center_end = 0.40, 0.60
    center_mask = (x_dense > center_start) & (x_dense < center_end)

    coeffs = np.polyfit(
        x_dense[center_mask],
        yc_base[center_mask],
        1
    )
    yc_trend = np.polyval(coeffs, x_dense[center_mask])

    blend_x = (x_dense[center_mask] - center_start) / (center_end - center_start)
    weights = 0.5 * (1 - np.cos(np.pi * blend_x)) * 0.6

    yc_new[center_mask] = (1 - weights) * yc_new[center_mask] + weights * yc_trend
    yc_new = gaussian_filter1d(yc_new, sigma=2.0)

    xu, yu, xl, yl = compute_airfoil(x_dense, yc_new, yt_base)

    # Plot airfoil surfaces
    plt.plot(xu, yu, color=color, lw = 1.5, label=aoa_folder)
    plt.plot(xl, yl, color=color, lw = 1.5)

    # # Plot camber line (black dashed, no legend)
    # plt.plot(x_dense, yc_new, color='black', linestyle='--', linewidth=0.5, alpha=1)
    found_any = True

    cl, cd, cm = apply_gb_correction(delta_y, xu, yu, xl, yl, AoA)
    aoa_vals.append(AoA)
    cl_vals.append(cl)
    cd_vals.append(cd)
    cm_vals.append(cm)
    ld_vals.append(cl/cd)
    geom_data.append((AoA, xu, yu, xl, yl))

    # ================================
    # LOAD POLAR.DAT (XFOIL single AoA)
    # ================================
    polar_file = os.path.join(re_dir, aoa_folder, "polar.dat")

    if os.path.exists(polar_file):
        try:
            pcl, pcd, pcm = parse_polar(polar_file)
            polar_cl_vals.append(pcl)
            polar_cd_vals.append(pcd)
            polar_cm_vals.append(pcm)
            polar_ld_vals.append(pcl / max(pcd, 1e-6))
        except:
            print(f"⚠️ Failed to read polar.dat at AoA {AoA:.2f}")
            polar_cl_vals.append(np.nan)
            polar_cd_vals.append(np.nan)
            polar_cm_vals.append(np.nan)
            polar_ld_vals.append(np.nan)
    else:
        print(f"⚠️ Missing polar.dat at AoA {AoA:.2f}")
        polar_cl_vals.append(np.nan)
        polar_cd_vals.append(np.nan)
        polar_cm_vals.append(np.nan)
        polar_ld_vals.append(np.nan)

    print(f"AoA {AoA:.2f}° → CL={cl:.6f} CD={cd:.6f} CM={cm:.6f} CL/CD={cl/cd:.2f}")

for aoa_folder in aoa_folders:
    AoA = float(aoa_folder.split()[0])
    cp_file = os.path.join(re_dir, aoa_folder, "cp.dat")

    if not os.path.exists(cp_file):
        print(f"⚠️ Missing Cp file at AoA {AoA:.2f}")
        continue

    try:
        x_cp, cp = parse_cp(cp_file)
        cp_data.append((AoA, x_cp, cp))
    except:
        print(f"⚠️ Failed to read Cp at AoA {AoA:.2f}")

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

xfoil_file = os.path.join("XFOIL Results", f"NACA2412Re{re_folder}.txt")

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
aoa_arr = np.array(aoa_vals)
ld_arr = np.array(polar_ld_vals)

mask = ~np.isnan(ld_arr)
idx = np.argsort(aoa_arr[mask])

plt.plot(
    aoa_arr[mask][idx],
    ld_arr[mask][idx],
    'd--',
    lw=2,
    label="XFOIL (per-airfoil)",
    color="tab:green"
)
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
aoa_arr = np.array(aoa_vals)
cm_arr = np.array(polar_cm_vals) / 4

mask = ~np.isnan(cm_arr)

plt.plot(
    aoa_arr[mask],
    cm_arr[mask],
    'd--',
    lw=2,
    label="XFOIL (per-airfoil)",
    color="tab:green"
)

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

# ================================
# PLOT CL/CD vs CM (fixed colorbar)
# ================================
plt.figure(figsize=(10,6))

# NEAT Airfoil points
sc_neat = plt.scatter(cm_vals, ld_vals,
                      c=aoa_vals, cmap='viridis', s=60, edgecolor='k', label="NEAT Airfoil")

# XFOIL per-AoA points (from polar.dat)
cm_arr = np.array(polar_cm_vals) / 4
ld_arr = np.array(polar_ld_vals)
mask = ~np.isnan(cm_arr) & ~np.isnan(ld_arr)
sc_xfoil = plt.scatter(cm_arr[mask], ld_arr[mask],
                       c=aoa_arr[mask], cmap='viridis', marker='d', s=60, label="XFOIL per-airfoil")

# XFOIL NACA 2412 curve
plt.plot(xfoil_filtered["Cm"], xfoil_filtered["CL_CD"],
         's--', color='tab:orange', lw=2, label="NACA2412 XFOIL")

plt.grid(True)
plt.xlabel("Pitching Moment Coefficient (CM)", fontweight='bold')
plt.ylabel("Lift-to-Drag Ratio (CL/CD)", fontweight='bold')
plt.title(f"CL/CD vs CM @ Re={re_folder}", fontsize=16, fontweight='bold')

# Colorbar based on NEAT scatter
cbar = plt.colorbar(sc_neat)
cbar.set_label("Angle of Attack (°)", fontweight='bold')

plt.legend()
plt.tight_layout()
plt.show()

# ================================
# Cp OVERLAY PLOT
# ================================
plt.figure(figsize=(10,6))

for i, (AoA, x_cp, cp) in enumerate(cp_data):
    color = plt.cm.viridis(i / len(cp_data))

    le = np.argmin(x_cp)

    plt.plot(x_cp[:le+1], cp[:le+1], color=color)
    plt.plot(x_cp[le:], cp[le:], color=color, linestyle='--')

plt.gca().invert_yaxis()
plt.grid(True)
plt.xlabel("x/c", fontweight='bold')
plt.ylabel("Cp", fontweight='bold')
plt.title("Cp(x) Overlay Across AoA", fontsize=16, fontweight='bold')

sm = plt.cm.ScalarMappable(
    cmap="viridis",
    norm=mpl.colors.Normalize(vmin=min(aoa_vals), vmax=max(aoa_vals))
)
sm.set_array([])

# 🔥 FIX HERE
cbar = plt.colorbar(sm, ax=plt.gca(),  pad=0.05)
cbar.set_label("Angle of Attack (°)", fontweight='bold')

plt.tight_layout()
plt.show()

# ================================
# STACKED Cp + GEOMETRY
# ================================
fig = plt.figure(figsize=(12,8))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.03], height_ratios=[1,1], hspace=0.15)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0], sharex=ax1)

# --- Cp ---
for i, (AoA, x_cp, cp) in enumerate(cp_data):
    color = plt.cm.viridis(i / len(cp_data))
    le = np.argmin(x_cp)
    ax1.plot(x_cp[:le+1], cp[:le+1], color=color)
    ax1.plot(x_cp[le:], cp[le:], color=color, linestyle='--')

ax1.invert_yaxis()
ax1.set_ylabel("Cp", fontweight='bold')
ax1.set_title("Cp Distribution Overlay", fontweight='bold')
ax1.grid(True)

# --- Geometry ---
for i, (AoA, x_cp, cp) in enumerate(cp_data):
    color = plt.cm.viridis(i / len(cp_data))
    match = next((g for g in geom_data if abs(g[0]-AoA)<1e-6), None)
    if match is None:
        continue
    _, xu, yu, xl, yl = match
    ax2.plot(xu, yu, color=color)
    ax2.plot(xl, yl, color=color)

ax2.set_aspect('equal', 'box')
ax2.set_xlabel("x/c", fontweight='bold')
ax2.set_ylabel("y/c", fontweight='bold')
ax2.set_title("Airfoil Geometry Overlay", fontweight='bold')
ax2.grid(True)
ax2.set_ylim(-0.1, 0.15)  # <-- expand geometry vertical range

# --- Colorbar ---
sm = mpl.cm.ScalarMappable(cmap="viridis",
                           norm=mpl.colors.Normalize(vmin=min(aoa_vals), vmax=max(aoa_vals)))
sm.set_array([])
# Use add_axes to position manually: [left, bottom, width, height] in figure fraction
cax = fig.add_axes([0.85, 0.17, 0.025, 0.71])  # closer, thin, spans both subplots
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Angle of Attack (°)", fontweight='bold')

plt.show()

# ================================
# CONTROL POINT Δy ANALYSIS
# ================================

# Sort by AoA to keep plots clean
delta_y_data.sort(key=lambda x: x[0])

aoa_sorted = np.array([x[0] for x in delta_y_data])
delta_matrix = np.array([x[1] for x in delta_y_data])  # shape: (N_aoa, num_ctrl)

# -------------------------------
# 1. PHYSICAL Δy (actual geometry)
# -------------------------------
plt.figure(figsize=(10, 8))

for i in range(num_ctrl):
    plt.plot(
        aoa_sorted,
        delta_matrix[:, i],
        'o-',
        label=f"CP {i+1}"
    )

# Plot max bounds for reference
for i in range(num_ctrl):
    plt.plot(aoa_sorted, [ max_offsets[i]]*len(aoa_sorted), 'k--', alpha=0.15)
    plt.plot(aoa_sorted, [-max_offsets[i]]*len(aoa_sorted), 'k--', alpha=0.15)

plt.grid(True)
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("Δy (chord units)", fontweight='bold')
plt.title("Control Point Physical Δy vs AoA", fontsize=16, fontweight='bold')

plt.legend(loc="lower right", ncol=2)
plt.tight_layout()
plt.show()


# ------------------------------------
# 2. NORMALIZED Δy (control usage)
# ------------------------------------
plt.figure(figsize=(10, 8))

for i in range(num_ctrl):
    plt.plot(
        aoa_sorted,
        delta_matrix[:, i] / max_offsets[i],
        'o-',
        label=f"CP {i+1}"
    )

# Reference saturation lines
plt.axhline(1.0, color='k', linestyle='--', alpha=0.3)
plt.axhline(-1.0, color='k', linestyle='--', alpha=0.3)

plt.grid(True)
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("Normalized Δy / max_offset", fontweight='bold')
plt.title("Control Point Usage vs AoA", fontsize=16, fontweight='bold')

plt.legend(loc="lower right", ncol=2)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.imshow(
    delta_matrix.T,
    aspect='auto',
    origin='lower',
    extent=[aoa_sorted.min(), aoa_sorted.max(), 1, num_ctrl]
)

plt.colorbar(label="Δy (chord units)")
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("Control Point Index", fontweight='bold')
plt.title("Control Point Δy Heatmap vs AoA", fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

normalized = delta_matrix / max_offsets  # shape: (num_aoa, num_ctrl)

# Perform PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(normalized)

# Optional: flip PC1 if you want a specific reference shape
# For example, if the first row of your data should correspond to negative PC1:
if reduced[0, 0] > 0:  # flip condition depends on your reference
    reduced[:, 0] *= -1
    pca.components_[0] *= -1  # also flip the component itself if needed

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=aoa_sorted, cmap='viridis')
plt.colorbar(label="AoA (°)")

plt.xlabel("PC1 (Airfoil Shape)", fontweight='bold')
plt.ylabel("PC2 (Pitching Moment Control)", fontweight='bold')
plt.title("Control Strategy Manifold", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# print("PC1:", pca.components_[0])
# print("PC2:", pca.components_[1])
#
# plt.figure(figsize=(8,4))
# for i in range(num_ctrl):
#     plt.bar(i+1, pca.components_[0][i])
# plt.xlabel("Control Point Index")
# plt.ylabel("PC1 Weight")
# plt.title("Contribution of Control Points to PC1")
# plt.show()
#
# mode_scale = 1.0  # adjust for visualization
# for i, pc in enumerate(pca.components_[:2]):  # PC1 and PC2
#     plt.figure()
#     delta_mode = pc * mode_scale * max_offsets  # scale back to Δy
#     plt.plot(range(1, num_ctrl+1), delta_mode, 'o-')
#     plt.xlabel("Control Point Index")
#     plt.ylabel("Δy (chord units)")
#     plt.title(f"PC{i+1} Mode Shape")
#     plt.grid(True)
#     plt.show()