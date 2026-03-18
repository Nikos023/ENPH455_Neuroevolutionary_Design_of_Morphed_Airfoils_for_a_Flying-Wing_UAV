#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COMBINED NEAT AIRFOIL TESTER
- Training-consistent geometry
- NeuralFoil + GB correction
- XFOIL validation
- Cp plotting
"""

import neat
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import os
import joblib
import subprocess
from neuralfoil import get_aero_from_coordinates

# ================== USER SETTINGS ==========================
REYNOLDS = 1e6
AoA = 0.00

xfoil_path = "/Users/nicholasburen/Downloads/xfoil/bin/xfoil"

re_folder = f"{REYNOLDS:.0e}".replace("+0", "").replace("+", "")
config_path = "NEAT Config Single Genome.ini"

aoa_folder = f"{AoA:.2f} Degrees"
save_dir = os.path.join("BestGenomes", f"Re{re_folder}", aoa_folder)
os.makedirs(save_dir, exist_ok=True)

best_genome_file = os.path.join(save_dir, "best_genome_nf.pkl")
output_name = "NEAT_airfoil"

# ================== AIRFOIL SETUP ==========================
num_ctrl = 10
n_each_side = num_ctrl // 2
x_ctrl = np.concatenate([
    np.linspace(0, 1/3, n_each_side, endpoint=False),
    np.linspace(2/3, 1, n_each_side)
])

num_points = 1000
beta = np.linspace(0, np.pi, num_points)
x_dense = (1 - np.cos(beta)) / 2

m, p, t = 0.02, 0.4, 0.12

yt_base = 5 * t * (
    0.2969*np.sqrt(x_dense)
    - 0.126*x_dense
    - 0.3516*x_dense**2
    + 0.2843*x_dense**3
    - 0.1015*x_dense**4
)

# Max delta_y per control point
max_offsets = np.array([0.12,0.10,0.08,0.02,0.001,0.001,0.02,0.08,0.10,0.12])
max_offsets = max_offsets * 0.75

# ================== LOAD GB MODELS =========================
model_dir = os.path.join("../Comparison/Comparison Results/global_model/2000gb", f"Re{re_folder}")
model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb.joblib"))

# ================== HELPER FUNCTIONS =======================
def yc_base_function(x):
    return np.where(
        x < p,
        m / p**2 * (2*p*x - x**2),
        m / (1 - p)**2 * ((1 - 2*p) + 2*p*x - x**2)
    )

def y_ctrl_base_function():
    return np.interp(x_ctrl, x_dense, yc_base_function(x_dense))

def smooth_camber(x_ctrl, y_ctrl):
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

# ================== GB CORRECTION ==========================
def apply_gb_correction(delta_y, xu, yu, xl, yl):
    dy_vec = delta_y
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)

    X_input = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2, AoA]).reshape(1, -1)

    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

    aero = get_aero_from_coordinates(
        coordinates=coords,
        alpha=[AoA],
        Re=REYNOLDS,
        model_size="xxxlarge"
    )

    cl_nf, cd_nf, cm_nf = aero["CL"][0], aero["CD"][0], aero["CM"][0]

    if (AoA >= 0):
        cl = cl_nf
        cd = max(cd_nf, 1e-4)
        cm = cm_nf

    else:
        cl = cl_nf - model_cl.predict(X_input)[0]
        cd = max(cd_nf - model_cd.predict(X_input)[0], 1e-4)
        cm = cm_nf - model_cm.predict(X_input)[0]



    return cl, cd, cm

# ================== XFOIL ================================
def run_xfoil(dat_file):
    cp_file = os.path.join(save_dir, "cp.dat")
    polar_file = os.path.join(save_dir, "polar.dat")

    # Clean old files
    if os.path.exists(cp_file): os.remove(cp_file)
    if os.path.exists(polar_file): os.remove(polar_file)

    commands = f"""
LOAD {dat_file}
PANE
OPER
VISC {REYNOLDS}
ITER 200
PACC
{polar_file}

ALFA {AoA}
CPWR {cp_file}

PACC

QUIT
"""

    process = subprocess.run(
        [xfoil_path],
        input=commands,
        text=True,
        capture_output=True
    )

    if process.returncode != 0:
        print("❌ XFOIL failed")
        print(process.stderr)
        return None, None

    # 🔥 Critical: verify files exist
    if not os.path.exists(cp_file):
        print("⚠️ Cp file not generated")
        return None, polar_file

    return cp_file, polar_file

def parse_polar(file):
    data = np.loadtxt(file, skiprows=12)
    if data.ndim == 1: data = data.reshape(1,-1)
    return data[0,1], data[0,2], data[0,4]

def parse_cp(file):
    data = np.loadtxt(file, skiprows=3)
    return data[:,0], data[:,1]

def plot_cp(x, cp):
    le = np.argmin(x)
    plt.figure()
    plt.plot(x[:le+1], cp[:le+1], label="Upper")
    plt.plot(x[le:], cp[le:], label="Lower")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid()
    plt.title("Cp Distribution")
    plt.show()

# ================== LOAD GENOME ============================
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

with open(best_genome_file, "rb") as f:
    genome = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(genome, config)

# ================== BUILD AIRFOIL ==========================
y_ctrl_base = y_ctrl_base_function()
X_input = np.hstack([y_ctrl_base, AoA]).reshape(1, -1)

raw = np.array(net.activate(X_input.flatten()))[:num_ctrl]
delta_y = np.clip(raw * max_offsets * 2, -max_offsets, max_offsets)
delta_y = gaussian_filter1d(delta_y, sigma=2)

y_ctrl_new = y_ctrl_base + delta_y
yc_new = smooth_camber(x_ctrl, y_ctrl_new)

# center locking
mask = (x_dense > 0.33) & (x_dense < 0.55)
trend = np.polyval(
    np.polyfit(x_dense[mask], yc_base_function(x_dense)[mask], 1),
    x_dense[mask]
)

blend = (x_dense[mask]-0.33)/(0.55-0.45)
weights = 0.5*(1-np.cos(np.pi*blend))*0.6

yc_new[mask] = (1-weights)*yc_new[mask] + weights*trend
yc_new = gaussian_filter1d(yc_new, sigma=2)

xu, yu, xl, yl = compute_airfoil(x_dense, yc_new, yt_base)

# ================== SAVE GEOMETRY ==========================
dat_file = os.path.join(save_dir, f"{output_name}.dat")

beta = np.linspace(0, np.pi, 151)
x_cos = 0.5*(1-np.cos(beta))

yU = np.interp(x_cos, xu, yu)
yL = np.interp(x_cos, xl, yl)

x_all = np.concatenate([x_cos[::-1], x_cos[1:]])
y_all = np.concatenate([yU[::-1], yL[1:]])

with open(dat_file, "w") as f:
    f.write(f"{output_name}\n")
    for xi, yi in zip(x_all, y_all):
        f.write(f"{xi:.8f} {yi:.8f}\n")

print(f"✅ Geometry saved: {dat_file}")

# ================== MODEL PREDICTION =======================
cl_m, cd_m, cm_m = apply_gb_correction(delta_y, xu, yu, xl, yl)

print("\n📊 NeuralFoil + GB Correction")
print(f"CL = {cl_m:.6f}, CD = {cd_m:.6f}, CM = {cm_m:.6f}, CL/CD = {cl_m/cd_m:.6f}")

cp_file, polar_file = run_xfoil(dat_file)

cl_x, cd_x, cm_x = parse_polar(polar_file)

print("📊 XFOIL Results")
print(f"CL = {cl_x:.6f}, CD = {cd_x:.6f}, CM = {cm_x:.6f}, CL/CD = {cl_x/cd_x:.6f}")

if cp_file and os.path.exists(cp_file):
    x_cp, cp = parse_cp(cp_file)
    plot_cp(x_cp, cp)
else:
    print("⚠️ Skipping Cp plot (file missing)")

# ================== STACKED AIRFOIL + Cp PLOTS ==================
if cp_file and os.path.exists(cp_file):
    x_cp, cp = parse_cp(cp_file)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    fig.suptitle(f"Airfoil Analysis (NEAT) at AoA = {AoA}°", fontsize=16, fontweight='bold')

    # --- Cp plot on top ---
    le = np.argmin(x_cp)  # leading edge index
    ax1.plot(x_cp[:le + 1], cp[:le + 1], color='C0', label="Upper Surface")
    ax1.plot(x_cp[le:], cp[le:], color='C1', label="Lower Surface")
    ax1.invert_yaxis()
    ax1.set_ylabel("Cp", fontweight='bold')
    ax1.set_title("Pressure Coefficient Distribution", fontsize=14, fontweight='bold')
    ax1.grid(True)
    ax1.legend()

    # --- Airfoil geometry below ---
    ax2.plot(xu, yu, color='C0', label="Upper Surface")
    ax2.plot(xl, yl, color='C1', label="Lower Surface")
    ax2.plot(x_dense, yc_new, color='black', linestyle='--', label="Camber Line")
    ax2.set_aspect('equal', 'box')
    ax2.set_xlabel("x (chord)", fontweight='bold')
    ax2.set_ylabel("y", fontweight='bold')
    ax2.set_title(f"NEAT Airfoil Geometry", fontsize=14, fontweight='bold')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Skipping stacked plot (Cp file missing)")

# ================== PLOTS ==================
# 1️⃣ Airfoil surfaces + camber line
plt.figure(figsize=(12, 6))
plt.plot(xu, yu, 'b-', lw=2, label="Upper Surface")
plt.plot(xl, yl, 'b-', lw=2, label="Lower Surface")
plt.plot(x_dense, yc_new, 'r--', lw=1.5, label="Camber Line")
plt.axis("equal")
plt.grid(True)
plt.xlabel("x (chord)")
plt.ylabel("y")
plt.title(f"NEAT Airfoil @ AoA = {AoA}°")
plt.legend()
plt.show()

# 2️⃣ Airfoil physically rotated to AoA
theta = -np.deg2rad(AoA)
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

# 3️⃣ Overlay NEAT airfoil with NACA 2412
yc_naca = yc_base_function(x_dense)
naca_file = "../Morphing/airfoil_xfoil_2412.dat"
if os.path.exists(naca_file):
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
else:
    print("⚠️ NACA 2412 reference file not found, skipping overlay plot")

# ================== NEURALFOIL AIRFOIL ===================
coords_nf = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)
aero_nf = get_aero_from_coordinates(
    coordinates=coords_nf,
    alpha=[AoA],
    Re=REYNOLDS,
    model_size="xxxlarge"
)

# Extract NeuralFoil CL/CD/CM if needed
cl_nf, cd_nf, cm_nf = aero_nf["CL"][0], aero_nf["CD"][0], aero_nf["CM"][0]
print("\n📊 NeuralFoil Raw Prediction")
print(f"CL = {cl_nf:.6f}, CD = {cd_nf:.6f}, CM = {cm_nf:.6f}, CL/CD = {cl_nf/cd_nf:.6f}")

# Coordinates for plotting
# NeuralFoil usually returns the same points, so we can use coords_nf directly
xu_nf, yu_nf = coords_nf[:num_points,0], coords_nf[:num_points,1]
xl_nf, yl_nf = coords_nf[num_points-1:,0], coords_nf[num_points-1:,1]