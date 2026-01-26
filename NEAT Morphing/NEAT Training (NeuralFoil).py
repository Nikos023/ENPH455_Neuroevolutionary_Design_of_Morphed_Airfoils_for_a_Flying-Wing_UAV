#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEAT Airfoil Trainer with NeuralFoil + GB Correction (CM-priority)
- Primary objective: minimize pitching moment (CM ~ 0)
- Secondary objective: maximize CL/CD
- Evaluates each genome online with NeuralFoil + GB correction
"""

import neat
import numpy as np
import neuralfoil as nf
import pickle
import os
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import joblib
from neuralfoil import get_aero_from_coordinates

# ================== BASE PARAMETERS =========================
m, p, t = 0.02, 0.4, 0.12
num_points = 1000
num_ctrl = 10
REYNOLDS = 1e6

# Cosine spacing
beta = np.linspace(0, np.pi, num_points)
x_dense = (1 - np.cos(beta)) / 2

# Base camber
yc_base = np.where(
    x_dense < p,
    m / p**2 * (2 * p * x_dense - x_dense**2),
    m / (1 - p)**2 * ((1 - 2*p) + 2*p*x_dense - x_dense**2)
)

# Base thickness
yt_base = 5 * t * (0.2969*np.sqrt(x_dense) - 0.126*x_dense - 0.3516*x_dense**2 +
                    0.2843*x_dense**3 - 0.1015*x_dense**4)

# Control points
n_each_side = num_ctrl // 2
x_ctrl_left = np.linspace(0, 1/3, n_each_side, endpoint=False)
x_ctrl_right = np.linspace(2/3, 1, n_each_side)
x_ctrl = np.concatenate([x_ctrl_left, x_ctrl_right])
y_ctrl_base = np.interp(x_ctrl, x_dense, yc_base)

# ================== HELPER FUNCTIONS ========================
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
    coords = np.vstack([coords_upper, coords_lower[1:]])
    return coords

# ================== LOAD GB MODELS ==========================
model_dir = "../Comparison/Comparison Results/global_model"
model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb_1000_samples.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb_1000_samples.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb_1000_samples.joblib"))

# ================== FITNESS FUNCTION =======================
def compute_fitness(genome, config, target_aoa):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    y_ctrl_noisy = y_ctrl_base + np.random.normal(0, 0.001, size=num_ctrl)
    X_input_net = np.hstack([y_ctrl_noisy, target_aoa]).reshape(1, -1)

    # Network outputs delta_y
    raw_output = np.array(net.activate(X_input_net.flatten()))[:num_ctrl]
    max_offsets = np.array([0.05,0.04,0.03,0.02,0.01,0.01,0.02,0.03,0.04,0.05])
    delta_y = np.clip(raw_output * max_offsets * 2.0, -0.05, 0.05)

    y_ctrl = y_ctrl_base + delta_y
    yc = smooth_camber(x_ctrl, y_ctrl, x_dense)
    xu, yu, xl, yl = compute_airfoil(x_dense, yc, yt_base)
    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

    # --- NeuralFoil Evaluation ---
    try:
        aero = get_aero_from_coordinates(
            coordinates=coords,
            alpha=[target_aoa],
            Re=REYNOLDS,
            model_size="xxxlarge",
            n_crit=9.0,
            xtr_upper=1.0,
            xtr_lower=1.0
        )
        cl_nf = aero["CL"][0]
        cd_nf = aero["CD"][0]
        cm_nf = aero["CM"][0]
    except Exception as e:
        print(f"ERROR: NeuralFoil evaluation failed: {e}")
        return -5.0

    # GB correction
    dy_vec = delta_y
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)
    X_input_gb = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2, target_aoa]).reshape(1, -1)
    cl_corr = cl_nf - model_cl.predict(X_input_gb)[0]
    cd_corr = max(cd_nf - model_cd.predict(X_input_gb)[0], 1e-5)
    cm_corr = cm_nf - model_cm.predict(X_input_gb)[0]

    # ================= PRIORITIZE CM ==========================
    fitness = np.exp(-50 * abs(cm_corr)) * (1 + cl_corr / cd_corr)

    # ===================== GEOMETRY PENALTIES =================
    smooth_penalty = np.sum(np.maximum(0, np.abs(np.diff(delta_y)) - 0.05)**2)
    fitness -= 0.2 * smooth_penalty

    d2yc_dx2 = np.gradient(np.gradient(yc, x_dense), x_dense)
    center_mask = (x_dense > 0.45) & (x_dense < 0.55)
    curvature_violation = np.maximum(0, np.abs(d2yc_dx2[center_mask]) - 0.8)
    fitness -= 1.5 * np.mean(curvature_violation**2)

    fitness -= 0.1 * (np.std(delta_y[:5]) + np.std(delta_y[5:]))

    d2yu_dx2 = np.gradient(np.gradient(yu, x_dense), x_dense)
    d2yl_dx2 = np.gradient(np.gradient(yl, x_dense), x_dense)
    mask = (x_dense > 0.2) & (x_dense < 0.8)
    fitness -= 0.8 * (np.mean(d2yu_dx2[mask]**2) + np.mean(d2yl_dx2[mask]**2))

    fitness += 0.05 * np.sum(np.abs(delta_y))

    return fitness

# ================== NEAT TRAINING ===========================
def train_for_aoa(target_aoa, generations=50):
    config_path = "NEAT Config Single Genome.ini"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())

    def eval_genomes(genomes, config):
        for gid, genome in genomes:
            genome.fitness = compute_fitness(genome, config, target_aoa)

    winner = pop.run(eval_genomes, generations)

    os.makedirs("BestGenomes", exist_ok=True)
    with open(f"BestGenomes/best_genome_nf_aoa{int(target_aoa)}.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("✅ Training complete!")

# ================== RUN TRAINING ===========================
train_for_aoa(5.0, generations=10)