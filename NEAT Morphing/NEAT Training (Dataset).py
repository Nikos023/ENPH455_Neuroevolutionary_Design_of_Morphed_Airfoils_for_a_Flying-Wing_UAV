#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEAT Airfoil Trainer with NeuralFoil + GB Correction
- Trains ONE AoA per run
- Produces AoA-specific geometries
- Saves best genome per AoA
"""

import neat
import numpy as np
import pickle
import os
import glob
from scipy.interpolate import interp1d, make_interp_spline
from scipy.ndimage import gaussian_filter1d
import joblib

# ================== CONFIGURATION ===========================
geom_dir = "../Morphing/Geometry/"
nf_dir = "../NeuralFoil/Simulation Results/"
comparison_dir = "../Comparison/Comparison Results"
model_dir = "../Comparison/Comparison Results/global_model"
os.makedirs(comparison_dir, exist_ok=True)

num_ctrl = 10
n_each_side = num_ctrl // 2
x_ctrl_left = np.linspace(0, 1 / 3, n_each_side, endpoint=False)
x_ctrl_right = np.linspace(2 / 3, 1, n_each_side)
x_ctrl = np.concatenate([x_ctrl_left, x_ctrl_right])

num_points = 1000
beta = np.linspace(0, np.pi, num_points)
x_dense = (1 - np.cos(beta)) / 2

# Base NACA 4-digit airfoil
m, p, t = 0.02, 0.4, 0.12
yt_base = 5 * t * (
    0.2969*np.sqrt(x_dense)
    - 0.1260*x_dense
    - 0.3516*x_dense**2
    + 0.2843*x_dense**3
    - 0.1015*x_dense**4
)

delta_y_max = 0.05
TRAIN_AOA = None   # <-- set per training run

# ================== HELPER FUNCTIONS ========================

def yc_base_function(x):
    return np.where(
        x < p,
        m/p**2 * (2*p*x - x**2),
        m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2)
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

def read_geometry_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    idx_ctrl = [i for i, l in enumerate(lines) if '=== Control Points' in l][0] + 1
    ctrl_lines = []
    for line in lines[idx_ctrl:]:
        if '===' in line:
            break
        line = line.strip()
        if line:
            ctrl_lines.append(line)
    ctrl_points = np.array([[float(v) for v in l.split(',')] for l in ctrl_lines])
    return ctrl_points[:, 0], ctrl_points[:, 1]

def read_polar(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    skiprows = 5
    data = np.loadtxt(filename, skiprows=skiprows)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

# ================== LOAD GLOBAL MODELS =====================
model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb_2000_samples.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb_2000_samples.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb_2000_samples.joblib"))

# ================== FITNESS FUNCTION =======================
def compute_fitness(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    y_ctrl_base = y_ctrl_base_function()

    geom_files = sorted(glob.glob(os.path.join(geom_dir, "airfoil_points_*.txt")))
    x_base, y_base = read_geometry_file(geom_files[0])

    # ---------------- Feature construction ----------------
    dy_vec = (y_ctrl_base - y_base).astype(float)
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)
    base_features = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2])

    AoA = TRAIN_AOA
    X_input = np.hstack([base_features, AoA]).reshape(1, -1)

    # ---------------- Network output ----------------
    delta_y = np.array(net.activate(X_input.flatten()))[:num_ctrl]

    max_offsets = np.array([
        0.035, 0.03, 0.025, 0.02, 0.01,
        0.01, 0.02, 0.025, 0.03, 0.035
    ])
    delta_y *= max_offsets

    # ---------------- Geometry construction ----------------
    y_ctrl = y_ctrl_base + delta_y
    yc = smooth_camber(x_ctrl, y_ctrl, x_dense)
    xu, yu, xl, yl = compute_airfoil(x_dense, yc, yt_base)

    # ---------------- NeuralFoil baseline ----------------
    nf_files = sorted(glob.glob(os.path.join(nf_dir, "polar_NeuralFoil_*.txt")))
    alpha_nf, cl_nf, cd_nf, cm_nf = read_polar(nf_files[0])

    f_cl = interp1d(alpha_nf, cl_nf, fill_value="extrapolate")
    f_cd = interp1d(alpha_nf, cd_nf, fill_value="extrapolate")
    f_cm = interp1d(alpha_nf, cm_nf, fill_value="extrapolate")

    cl_nf_i = f_cl(AoA)
    cd_nf_i = f_cd(AoA)
    cm_nf_i = f_cm(AoA)

    err_cl = model_cl.predict(X_input)[0]
    err_cd = model_cd.predict(X_input)[0]
    err_cm = model_cm.predict(X_input)[0]

    cl_corr = cl_nf_i - err_cl
    cd_corr = cd_nf_i - err_cd
    cm_corr = cm_nf_i - err_cm

    # ---------------- Base aerodynamic fitness ----------------
    base_fitness = cl_corr / cd_corr - abs(cm_corr)
    fitness = base_fitness / (1.0 + abs(base_fitness))

    # ========================================================
    # ================ GEOMETRY-BASED PENALTIES ===============
    # ========================================================

    # ---- 1. Control smoothness (kept) ----
    smooth_penalty = np.sum(
        np.maximum(0, np.abs(np.diff(delta_y)) - 0.05)**2
    )
    fitness -= 0.2 * smooth_penalty

    # ---- 2. Fuselage attachment constraint (NEW, geometry-based) ----
    # Penalize excessive camber curvature near center chord
    d2yc_dx2 = np.gradient(np.gradient(yc, x_dense), x_dense)

    center_mask = (x_dense > 0.45) & (x_dense < 0.55)

    curvature_allowance = 0.8     # structural flexibility
    curvature_stiffness = 1.5     # attachment rigidity

    curvature_violation = np.maximum(
        0.0,
        np.abs(d2yc_dx2[center_mask]) - curvature_allowance
    )

    fitness -= curvature_stiffness * np.mean(curvature_violation**2)

    # ---- 3. Group coherence (kept) ----
    fitness -= 0.1 * (np.std(delta_y[:5]) + np.std(delta_y[5:]))

    # ---- 4. Global curvature control (kept) ----
    d2yu = np.gradient(np.gradient(yu, x_dense), x_dense)
    d2yl = np.gradient(np.gradient(yl, x_dense), x_dense)
    mask = (x_dense > 0.2) & (x_dense < 0.8)

    fitness -= 0.8 * (
        np.mean(d2yu[mask]**2) + np.mean(d2yl[mask]**2)
    )

    # ---- 5. Encourage meaningful morphing (kept) ----
    fitness += 0.05 * np.sum(np.abs(delta_y))

    return fitness

# ================== NEAT CONFIG ============================
config_path = "NEAT Config Single Genome.ini"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# ================== TRAINING FUNCTION ======================
def train_for_aoa(target_aoa, generations=100):
    global TRAIN_AOA
    TRAIN_AOA = target_aoa

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    def eval_genomes(genomes, config):
        for gid, genome in genomes:
            genome.fitness = compute_fitness(genome, config)

    winner = pop.run(eval_genomes, generations)

    fname = f"best_genome_data_aoa{int(target_aoa)}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(winner, f)

    print(f"✅ Saved {fname}")

# ================== RUN TWO INDEPENDENT TRAINS =============
train_for_aoa(5.0, generations=20)