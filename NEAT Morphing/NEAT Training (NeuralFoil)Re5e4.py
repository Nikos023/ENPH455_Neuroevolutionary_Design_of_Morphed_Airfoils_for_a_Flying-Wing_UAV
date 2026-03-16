#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEAT Airfoil Trainer with NeuralFoil + GB Correction (CM-priority, Smooth)
- Primary objective: minimize pitching moment (CM ~ 0)
- Secondary objective: maximize CL/CD
- Evaluates each genome online with NeuralFoil + GB correction
- Applies full smoothing of geometry
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
from Visualize import draw_net
import matplotlib.pyplot as plt

REYNOLDS = 5e4
re_folder = f"{REYNOLDS:.0e}".replace("+0", "").replace("+", "")
AoA = 5.0
Gen = 350
CURRENT_GEN = 0

# ================== BASE PARAMETERS =========================
m, p, t = 0.02, 0.4, 0.12
num_points = 1000
num_ctrl = 10

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

# Max delta_y per control point
#max_offsets = np.array([0.05,0.04,0.03,0.02,0.01,0.01,0.02,0.03,0.04,0.05])
#max_offsets = np.array([0.05,0.04,0.03,0.02,0.01,0.01,0.04,0.06,0.08,0.10])
max_offsets = np.array([0.07,0.06,0.05,0.03,0.01,0.01,0.06,0.09,0.12,0.14])
#max_offsets = np.array([0.08,0.06,0.04,0.04,0.01,0.01,0.03,0.04,0.06,0.08])
#max_offsets = np.array([0.10,0.08,0.06,0.04,0.01,0.01,0.04,0.06,0.08,0.10])
#max_offsets = np.array([0.15,0.12,0.09,0.06,0.01,0.01,0.06,0.09,0.12,0.15])
#max_offsets = np.array([0.20,0.16,0.12,0.09,0.01,0.01,0.09,0.012,0.16,0.20])

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

def get_cm_limit(gen):
    start = 0.02
    end = 0.001
    decay_gens = int(0.8 * Gen)

    progress = min(1.0, gen / decay_gens)
    smooth = 0.5 * (1 - np.cos(np.pi * progress))

    return start - (start - end) * smooth

# ================== LOAD GB MODELS ==========================
model_dir = os.path.join("../Comparison/Comparison Results/global_model/2000gb", f"Re{re_folder}")
model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb.joblib"))

# ================== FITNESS FUNCTION =======================
def compute_fitness(genome, config, target_aoa, noise_sigma=0.0005):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    y_ctrl_noisy = y_ctrl_base
    X_input_net = np.hstack([y_ctrl_noisy, target_aoa]).reshape(1, -1)

    raw_output = np.array(net.activate(X_input_net.flatten()))[:num_ctrl]
    delta_y = np.clip(raw_output * max_offsets * 2.0, -max_offsets, max_offsets)
    delta_y_smooth = gaussian_filter1d(delta_y, sigma=2.0)
    y_ctrl = y_ctrl_base + delta_y_smooth

    yc = smooth_camber(x_ctrl, y_ctrl, x_dense)

    center_start, center_end = 0.33, 0.66
    center_mask = (x_dense > center_start) & (x_dense < center_end)

    coeffs = np.polyfit(
        x_dense[center_mask],
        yc_base[center_mask],
        1
    )
    yc_trend = np.polyval(coeffs, x_dense[center_mask])

    blend_x = (x_dense[center_mask] - center_start) / (center_end - center_start)
    weights = 0.5 * (1 - np.cos(np.pi * blend_x))
    weights *= 0.6

    yc[center_mask] = (
        (1 - weights) * yc[center_mask] +
        weights * yc_trend
    )

    yc = gaussian_filter1d(yc, sigma=2.0)

    xu, yu, xl, yl = compute_airfoil(x_dense, yc, yt_base)
    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

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

    dy_vec = delta_y_smooth
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)
    X_input_gb = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2, target_aoa]).reshape(1, -1)

    cl_corr = cl_nf - model_cl.predict(X_input_gb)[0]
    cd_corr = max(cd_nf - model_cd.predict(X_input_gb)[0], 1e-3)
    cm_corr = cm_nf - model_cm.predict(X_input_gb)[0]

    CM_LIMIT = get_cm_limit(CURRENT_GEN)

    progress = min(1.0, CURRENT_GEN / Gen)

    if REYNOLDS >= 1e6:
        cm_weight = 140 * (1 - 0.30 * progress)
        ld_weight = 180 * (1 + 0.60 * progress)

    elif REYNOLDS >= 5e5:
        cm_weight = 160 * (1 - 0.30 * progress)
        ld_weight = 200 * (1 + 0.60 * progress)

    elif REYNOLDS >= 3e5:
        cm_weight = 180 * (1 - 0.30 * progress)
        ld_weight = 220 * (1 + 0.60 * progress)

    elif REYNOLDS >= 2e5:
        cm_weight = 200 * (1 - 0.30 * progress)
        ld_weight = 240 * (1 + 0.60 * progress)

    elif REYNOLDS >= 1e5:
        cm_weight = 230 * (1 - 0.30 * progress)
        ld_weight = 270 * (1 + 0.60 * progress)

    else:
        cm_weight = 260 * (1 - 0.30 * progress)
        ld_weight = 300 * (1 + 0.60 * progress)

    cm_violation = max(0.0, abs(cm_corr) - CM_LIMIT)
    cm_term = -cm_weight * (cm_violation / CM_LIMIT)

    # # Continuous penalty pushing CM → 0
    # cm_center_penalty = -cm_weight * (abs(cm_corr) / CM_LIMIT) ** 2
    #
    # # Strong wall if outside limit
    # cm_violation = max(0.0, abs(cm_corr) - CM_LIMIT)
    # cm_limit_penalty = -5 * cm_weight * (cm_violation / CM_LIMIT) ** 2
    #
    # cm_term = cm_center_penalty + cm_limit_penalty

    LD = cl_corr / cd_corr
    LD_safe = max(LD, 1e-3)

    ld_term = ld_weight * np.sqrt(LD_safe)

    fitness = cm_term + ld_term

    smooth_penalty = np.sum(np.maximum(0, np.abs(np.diff(delta_y_smooth)) - 0.05)**2)
    fitness -= 0.05 * smooth_penalty / num_ctrl

    d2yc_dx2 = np.gradient(np.gradient(yc, x_dense), x_dense)
    center_mask_penalty = (x_dense > 0.33) & (x_dense < 0.66)
    curvature_violation = np.maximum(0, np.abs(d2yc_dx2[center_mask_penalty]) - 0.8)
    fitness -= 0.3 * np.mean(curvature_violation ** 2)

    fitness -= 0.1 * (np.std(delta_y_smooth[:5]) + np.std(delta_y_smooth[5:])) / 0.05

    d2yu_dx2 = np.gradient(np.gradient(yu, x_dense), x_dense)
    d2yl_dx2 = np.gradient(np.gradient(yl, x_dense), x_dense)
    mask = (x_dense > 0.2) & (x_dense < 0.8)
    fitness -= 0.3 * (np.mean(d2yu_dx2[mask] ** 2) + np.mean(d2yl_dx2[mask] ** 2))

    return fitness

# ================== CM-ONLY EVALUATION ======================
def compute_cm_only(genome, config, target_aoa):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    X_input_net = np.hstack([y_ctrl_base, target_aoa]).reshape(1, -1)
    raw_output = np.array(net.activate(X_input_net.flatten()))[:num_ctrl]
    delta_y = np.clip(raw_output * max_offsets * 2.0, -max_offsets, max_offsets)
    delta_y_smooth = gaussian_filter1d(delta_y, sigma=2.0)
    y_ctrl = y_ctrl_base + delta_y_smooth

    yc = smooth_camber(x_ctrl, y_ctrl, x_dense)
    yc = gaussian_filter1d(yc, sigma=2.0)

    xu, yu, xl, yl = compute_airfoil(x_dense, yc, yt_base)
    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

    aero = get_aero_from_coordinates(
        coordinates=coords,
        alpha=[target_aoa],
        Re=REYNOLDS,
        model_size="xxxlarge",
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0
    )

    cm_nf = aero["CM"][0]

    dy_vec = delta_y_smooth
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)

    X_input_gb = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2, target_aoa]).reshape(1, -1)
    cm_corr = cm_nf - model_cm.predict(X_input_gb)[0]

    return cm_corr

# ================== CL/CD-ONLY EVALUATION ===================
# >>> ADDED
def compute_clcd_only(genome, config, target_aoa):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    X_input_net = np.hstack([y_ctrl_base, target_aoa]).reshape(1, -1)
    raw_output = np.array(net.activate(X_input_net.flatten()))[:num_ctrl]
    delta_y = np.clip(raw_output * max_offsets * 2.0, -max_offsets, max_offsets)
    delta_y_smooth = gaussian_filter1d(delta_y, sigma=2.0)
    y_ctrl = y_ctrl_base + delta_y_smooth

    yc = smooth_camber(x_ctrl, y_ctrl, x_dense)
    yc = gaussian_filter1d(yc, sigma=2.0)

    xu, yu, xl, yl = compute_airfoil(x_dense, yc, yt_base)
    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

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

    dy_vec = delta_y_smooth
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)

    X_input_gb = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2, target_aoa]).reshape(1, -1)

    cl_corr = cl_nf - model_cl.predict(X_input_gb)[0]
    cd_corr = max(cd_nf - model_cd.predict(X_input_gb)[0], 1e-3)

    return cl_corr / cd_corr

# ================== NEAT TRAINING ===========================
def train_for_aoa(target_aoa, generations=50):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "NEAT Config Single Genome.ini"
    )
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    def eval_genomes(genomes, config):
        global CURRENT_GEN
        # Evaluate fitness for each genome
        for gid, genome in genomes:
            genome.fitness = compute_fitness(genome, config, target_aoa)

        # Increment the generation counter after evaluating all genomes
        CURRENT_GEN += 1

    winner = pop.run(eval_genomes, generations)

    # ================== SAVE DIRECTORY STRUCTURE ==================
    aoa_folder = f"{int(target_aoa)} Degrees"

    save_dir = os.path.join("BestGenomes", f"Re{re_folder}", aoa_folder)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "best_genome_nf.pkl"), "wb") as f:
        pickle.dump(winner, f)

    draw_net(
        config,
        winner,
        view=True,
        filename=os.path.join(save_dir, "best_network")
    )

    # ================== CONVERGENCE PLOTS ==================

    cm_history = []
    clcd_history = []

    best_fitness = -np.inf
    best_genome_so_far = None

    for genome in stats.most_fit_genomes:

        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome_so_far = genome

        try:
            cm = compute_cm_only(best_genome_so_far, config, target_aoa)
            clcd = compute_clcd_only(best_genome_so_far, config, target_aoa)

            cm_history.append(abs(cm))
            clcd_history.append(clcd)

        except Exception:
            cm_history.append(np.nan)
            clcd_history.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # CM subplot
    ax1.plot(cm_history)
    ax1.set_ylabel("|Corrected CM|")
    ax1.set_title("Evolutionary Convergence for Reynolds 5e4")
    ax1.grid(True)

    # CL/CD subplot
    ax2.plot(clcd_history)
    ax2.set_ylabel("Corrected CL/CD")
    ax2.set_xlabel("Generation")
    ax2.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "training_convergence.png"),
        dpi=300
    )

    plt.show()

    print("✅ Training complete!")

# ================== RUN TRAINING ===========================
train_for_aoa(AoA, generations=Gen)
