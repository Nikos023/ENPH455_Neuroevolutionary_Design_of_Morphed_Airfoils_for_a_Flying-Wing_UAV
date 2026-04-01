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
plt.title(f"Cp(x) Overlay Across AoA @ Re={re_folder}", fontsize=16, fontweight='bold')

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

fig.suptitle(f"Cp(x) Overlay with Airfoil Geometry Across AoA @ Re={re_folder}", fontsize=16, fontweight='bold')
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
plt.title(f"Control Point Physical Δy vs AoA @ Re={re_folder}", fontsize=16, fontweight='bold')

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
plt.title(f"Control Point Usage vs AoA @ Re={re_folder}", fontsize=16, fontweight='bold')

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

cbar = plt.colorbar(label="Δy (chord units)")
cbar.set_label("Δy (chord units)", fontweight='bold')
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("Control Point Index", fontweight='bold')
plt.title(f"Control Point Δy Heatmap vs AoA @ Re={re_folder}", fontsize=16, fontweight='bold')

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
cbar = plt.colorbar(label="AoA (°)")
cbar.set_label("AoA (°)", fontweight='bold')

plt.xlabel("PC1 (Reflex / Moment Control)", fontweight='bold')
plt.ylabel("PC2 (Camber Deformation)", fontweight='bold')
plt.title(f"Control Strategy Manifold @ Re={re_folder}", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

aoa_mean = np.mean(aoa_sorted)
aoa_std = np.std(aoa_sorted)
aoa_norm = (aoa_sorted - aoa_mean) / aoa_std

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# -----------------------------
# Build spline models per CP
# -----------------------------
splines = []

for i in range(num_ctrl):
    cs = CubicSpline(
        aoa_sorted,
        delta_matrix[:, i],
        bc_type=((1, 0.0), (1, 0.0))  # zero slope at ends (often more stable)
    )
    splines.append(cs)

# -----------------------------
# Prediction function
# -----------------------------
def predict_delta(aoa):
    """
    Input:
        aoa : scalar or array of angles of attack (deg)

    Output:
        Δy : array shape (N, num_ctrl)
    """
    aoa = np.atleast_1d(aoa)

    # Clamp to avoid bad extrapolation
    aoa = np.clip(aoa, aoa_sorted.min(), aoa_sorted.max())

    deltas = np.zeros((len(aoa), num_ctrl))

    for i in range(num_ctrl):
        deltas[:, i] = splines[i](aoa)

    return deltas


# -----------------------------
# Example usage
# -----------------------------
aoa_test = 5.5
delta_pred = predict_delta(aoa_test)

print(f"AoA = {aoa_test}°")
for i, val in enumerate(delta_pred[0]):
    print(f"CP {i+1}: Δy = {val:.6f}")


# -----------------------------
# Plot spline fits
# -----------------------------
aoa_fine = np.linspace(aoa_sorted.min(), aoa_sorted.max(), 300)

# -----------------------------
# Compute R² for spline fits
# -----------------------------
r2_scores = []

for i in range(num_ctrl):
    y_true = delta_matrix[:, i]
    y_pred = splines[i](aoa_sorted)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    r2_scores.append(r2)

    print(f"CP {i+1}: R² = {r2:.6f}")

r2_scores = np.array(r2_scores)

print("\nAverage R²:", np.mean(r2_scores))
print("Min R²:", np.min(r2_scores))

plt.figure(figsize=(10, 8))

for i in range(num_ctrl):
    # Original data
    marker_plot, = plt.plot(
        aoa_sorted,
        delta_matrix[:, i],
        'o',
        markersize=5,
        label=f"CP {i + 1} Data"
    )
    color = marker_plot.get_color()

    # Spline fit
    fit_curve = splines[i](aoa_fine)

    plt.plot(
        aoa_fine,
        fit_curve,
        '-',
        linewidth=2,
        alpha=0.9,
        color=color,
        label=f"CP {i + 1} Spline (R²={r2_scores[i]:.4f})"
    )

    # Bounds
    plt.plot(aoa_fine, [max_offsets[i]] * len(aoa_fine), 'k--', alpha=0.1)
    plt.plot(aoa_fine, [-max_offsets[i]] * len(aoa_fine), 'k--', alpha=0.1)

plt.grid(True)
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("Δy (chord units)", fontweight='bold')
plt.title(
    f"Control Point Δy vs AoA (Cubic Spline Fit) @ Re={re_folder}",
    fontsize=16,
    fontweight='bold'
)
plt.legend(loc="lower right", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# -----------------------------
# Residuals plot
# -----------------------------
plt.figure(figsize=(10, 8))

for i in range(num_ctrl):
    y_pred = splines[i](aoa_sorted)
    residuals = delta_matrix[:, i] - y_pred

    plt.plot(
        aoa_sorted,
        residuals,
        'o',
        markersize=5,
        label=f"CP {i + 1}"
    )

plt.axhline(0, color='k', alpha=0.5)

plt.grid(True)
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("Residual Δy (chord units)", fontweight='bold')
plt.title(
    f"Residuals of Spline Fit @ Re={re_folder}",
    fontsize=16,
    fontweight='bold'
)
plt.legend(loc="upper right", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# ================================
# AIRFOIL RECONSTRUCTION COMPARISON
# ================================

def reconstruct_airfoil_from_delta(delta_y):
    y_ctrl_base = y_ctrl_base_function()
    y_ctrl_new = y_ctrl_base + delta_y

    yc_new = smooth_camber(x_ctrl, y_ctrl_new, x_dense)

    # --- center lock (same as training) ---
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

    return xu, yu, xl, yl


# -----------------------------
# Choose AoA to compare
# -----------------------------
aoa_test = 5.5  # change this

# --- TRUE (from NEAT dataset) ---
idx = np.argmin(np.abs(aoa_sorted - aoa_test))
aoa_true = aoa_sorted[idx]
delta_true = delta_matrix[idx]

# --- PREDICTED (from spline) ---
delta_pred = predict_delta(aoa_test)[0]

# --- Reconstruct both ---
xu_true, yu_true, xl_true, yl_true = reconstruct_airfoil_from_delta(delta_true)
xu_pred, yu_pred, xl_pred, yl_pred = reconstruct_airfoil_from_delta(delta_pred)


# -----------------------------
# Plot overlay
# -----------------------------
plt.figure(figsize=(10, 6))

# TRUE airfoil (NEAT)
plt.plot(xu_true, yu_true, 'k-', lw=2.5, label=f"True (NEAT) @ {aoa_true:.2f}°")
plt.plot(xl_true, yl_true, 'k-', lw=2.5)

# PREDICTED airfoil (Spline)
plt.plot(xu_pred, yu_pred, 'r--', lw=2, label=f"Spline @ {aoa_test:.2f}°")
plt.plot(xl_pred, yl_pred, 'r--', lw=2)

plt.axis("equal")
plt.grid(True)
plt.xlabel("x/c", fontweight='bold')
plt.ylabel("y/c", fontweight='bold')
plt.title(f"Airfoil Reconstruction Comparison @ Re={re_folder}", fontsize=16, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# Plot difference (error)
# -----------------------------
plt.figure(figsize=(10, 4))

# Interpolate onto same x grid (they already are, but keep safe)
error_upper = yu_pred - yu_true
error_lower = yl_pred - yl_true

plt.plot(x_dense, error_upper, label="Upper Surface Error")
plt.plot(x_dense, error_lower, label="Lower Surface Error")

plt.axhline(0, color='k', linestyle='--', alpha=0.5)

plt.grid(True)
plt.xlabel("x/c", fontweight='bold')
plt.ylabel("Δy Error", fontweight='bold')
plt.title("Reconstruction Error (Spline vs NEAT)", fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

# ================================
# CONTINUOUS SPLINE SWEEP (OFFSET AoA)
# ================================

aoa_sweep = np.arange(-5, 12.01, 0.25)
aoa_sweep_offset = aoa_sweep + 0.1  # <-- your offset

cl_spline_list = []
cd_spline_list = []
cm_spline_list = []
ld_spline_list = []

print("\nRunning spline-based continuous AoA sweep...")

for aoa in aoa_sweep_offset:

    # -----------------------------
    # Predict Δy from spline
    # -----------------------------
    delta = predict_delta(aoa)[0]

    # Clamp for safety
    delta = np.clip(delta, -max_offsets, max_offsets)

    # -----------------------------
    # Reconstruct airfoil
    # -----------------------------
    xu, yu, xl, yl = reconstruct_airfoil_from_delta(delta)

    # -----------------------------
    # Compute aero
    # -----------------------------
    cl, cd, cm = apply_gb_correction(delta, xu, yu, xl, yl, aoa)

    cl_spline_list.append(cl)
    cd_spline_list.append(cd)
    cm_spline_list.append(cm)
    ld_spline_list.append(cl / cd)

    print(f"AoA {aoa:.2f}° → CL={cl:.5f} CD={cd:.5f} CM={cm:.5f}")

# Convert to arrays
cl_spline_arr = np.array(cl_spline_list)
cd_spline_arr = np.array(cd_spline_list)
cm_spline_arr = np.array(cm_spline_list) / 4
ld_spline_arr = np.array(ld_spline_list)

# ================================
# PLOT: CL/CD COMPARISON
# ================================
plt.figure(figsize=(10,6))

# Original NEAT (discrete)
plt.plot(
    aoa_vals,
    ld_vals,
    'o-',
    lw=2,
    label="NEAT Optimization (Discrete)",
    color="tab:blue"
)

# Spline reconstruction (offset)
plt.plot(
    aoa_sweep_offset,
    ld_spline_arr,
    '-',
    lw=2.5,
    label="Spline Reconstruction (+0.1° offset)",
    color="tab:red"
)

plt.grid(True)
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("CL/CD", fontweight='bold')
plt.title(f"CL/CD Comparison (NEAT vs Spline) @ Re={re_folder}", fontsize=16, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.show()


# ================================
# PLOT: CM COMPARISON
# ================================
plt.figure(figsize=(10,6))

# Original NEAT
plt.plot(
    aoa_vals,
    cm_vals,
    'o-',
    lw=2,
    label="NEAT Optimization (Discrete)",
    color="tab:blue"
)

# Spline reconstruction
plt.plot(
    aoa_sweep_offset,
    cm_spline_arr,
    '-',
    lw=2.5,
    label="Spline Reconstruction (+0.1° offset)",
    color="tab:red"
)

plt.grid(True)
plt.xlabel("Angle of Attack (°)", fontweight='bold')
plt.ylabel("CM", fontweight='bold')
plt.title(f"CM Comparison (NEAT vs Spline) @ Re={re_folder}", fontsize=16, fontweight='bold')

plt.legend()
plt.tight_layout()
plt.show()

# ================================
# PERFORMANCE COMPARISON TABLE
# ================================
import pandas as pd

# Convert lists to arrays for convenience
aoa_arr = np.array(aoa_vals)
cl_neat = np.array(cl_vals)
cd_neat = np.array(cd_vals)
cm_neat = np.array(cm_vals)
ld_neat = cl_neat / cd_neat

cl_base = np.array(xfoil_filtered["Cl"])
cd_base = np.array(xfoil_filtered["Cd"])
cm_base = np.array(xfoil_filtered["Cm"])
ld_base = cl_base / cd_base

# Ensure AoA alignment
# Interpolate NEAT results at XFOIL AoA points
cl_neat_interp = np.interp(xfoil_filtered["Alpha"], aoa_arr, cl_neat)
cd_neat_interp = np.interp(xfoil_filtered["Alpha"], aoa_arr, cd_neat)
cm_neat_interp = np.interp(xfoil_filtered["Alpha"], aoa_arr, cm_neat)
ld_neat_interp = cl_neat_interp / cd_neat_interp

# Compute improvements
ld_improvement = 100 * (ld_neat_interp - ld_base) / ld_base
cm_delta = cm_neat_interp - cm_base

# Absolute Cm
cm_abs_base = np.abs(cm_base)
cm_abs_neat = np.abs(cm_neat_interp)

# ================================
# AVERAGE IMPROVEMENT vs NACA2412
# ================================
from scipy.interpolate import interp1d

# Interpolate NACA2412 XFOIL CL/CD and CM to NEAT AoA points
clcd_naca_interp = interp1d(
    xfoil_filtered["Alpha"],
    xfoil_filtered["CL_CD"],
    kind='linear',
    fill_value="extrapolate"
)(aoa_vals)

cm_naca_interp = interp1d(
    xfoil_filtered["Alpha"],
    xfoil_filtered["Cm"],
    kind='linear',
    fill_value="extrapolate"
)(aoa_vals)

# Ensure CL/CD is calculated as Cl / Cd directly (no percent scaling)
xfoil_filtered["CL_CD"] = xfoil_filtered["Cl"] / xfoil_filtered["Cd"]

# Interpolate NACA2412 CL/CD and CM to NEAT AoA points
clcd_naca_interp = np.interp(aoa_vals, xfoil_filtered["Alpha"], xfoil_filtered["CL_CD"])
cm_naca_interp   = np.interp(aoa_vals, xfoil_filtered["Alpha"], xfoil_filtered["Cm"])

# Compute improvement as relative to baseline
ld_vals_arr = np.array(ld_vals)
cm_vals_arr = np.array(cm_vals)

ld_improvement_pct = 100 * (ld_vals_arr - clcd_naca_interp) / clcd_naca_interp
cm_delta = cm_vals_arr - cm_naca_interp
cm_improvement_pct = 100 * cm_delta / np.abs(cm_naca_interp)  # absolute CM baseline

# Compute averages
avg_ld_base = np.mean(clcd_naca_interp)
avg_ld_neat = np.mean(ld_vals_arr)
avg_ld_improvement = ((avg_ld_neat - avg_ld_base)/ avg_ld_base) * 100

avg_cm_base = np.mean(cm_naca_interp)
avg_cm_neat = np.mean(cm_vals_arr)
avg_cm_delta = np.mean(cm_delta)
avg_cm_improvement = np.mean(cm_improvement_pct)

print("\n📊 AVERAGE PERFORMANCE VS NACA2412 @ Re={:.0e}".format(REYNOLDS))
print(f"Average CL/CD (NACA2412): {avg_ld_base:.5f}")
print(f"Average CL/CD (NEAT):      {avg_ld_neat:.5f}")
print(f"Average CL/CD Improvement (%): {avg_ld_improvement:.2f}%\n")

print(f"Average CM (NACA2412): {avg_cm_base:.5f}")
print(f"Average CM (NEAT):     {avg_cm_neat:.5f}")
print(f"Average ΔCM:           {avg_cm_delta:.5f}")
print(f"Average CM Improvement (%): {avg_cm_improvement:.2f}%")