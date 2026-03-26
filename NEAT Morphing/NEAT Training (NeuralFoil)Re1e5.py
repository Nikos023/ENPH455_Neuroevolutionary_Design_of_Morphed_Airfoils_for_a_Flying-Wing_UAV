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

REYNOLDS = 1e5
re_folder = f"{REYNOLDS:.0e}".replace("+0", "").replace("+", "")
AoA = 12.00
Gen = 400
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
max_offsets = np.array([0.12,0.10,0.08,0.04,0.01,0.01,0.02,0.08,0.10,0.12])
max_offsets = max_offsets * 0.65

# ================== HELPER FUNCTIONS ========================
def smooth_camber(x_ctrl, y_ctrl, x_dense):
    spline = make_interp_spline(x_ctrl, y_ctrl, k=1)
    yc = spline(x_dense)
    return gaussian_filter1d(yc, sigma=25)

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
    start = 0.005
    end = 0.0001
    decay_gens = int(0.9 * Gen)
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

    #Compute raw NEAT output
    raw_output = np.array(net.activate(X_input_net.flatten()))[:num_ctrl]

    #Convert to offsets (initially scaled by max_offsets)
    delta_y = raw_output * max_offsets * 2.0

    #Smooth all offsets
    delta_y_smooth = gaussian_filter1d(delta_y, sigma=2.0)

    #Enforce max_offsets on all points AFTER smoothing
    delta_y_smooth = np.clip(delta_y_smooth, -max_offsets, max_offsets)

    #Apply to base control points
    y_ctrl = y_ctrl_base + delta_y_smooth

    yc = smooth_camber(x_ctrl, y_ctrl, x_dense)

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
    yc[center_mask] = (1 - weights) * yc[center_mask] + weights * yc_trend

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

    cl_corr = cl_nf
    cd_corr = max(cd_nf, 1e-3)
    cm_corr = cm_nf - model_cm.predict(X_input_gb)[0]

    CM_LIMIT = get_cm_limit(CURRENT_GEN)
    progress = min(1.0, CURRENT_GEN / Gen)

    cm_weight = 1000 * (1 - 0.3 * progress)
    ld_weight = 340 * (1 + 0.7 * progress)

    cm_violation = max(0.0, abs(cm_corr) - CM_LIMIT)
    cm_term = -cm_weight * (cm_violation / CM_LIMIT)

    cd_safe = max(cd_corr, 1e-4)

    if cl_corr <= 0:
        efficiency = -(abs(cl_corr) / cd_safe) ** 2  # strongly punish negative lift
    else:
        efficiency = (cl_corr ** 1.4) / (cd_safe ** 1.5)

    ld_term = ld_weight * (efficiency / 60)

    # encourage low drag directly
    drag_penalty = 125 * cd_corr

    fitness = cm_term + ld_term - drag_penalty

    # discourage extreme lift spikes
    if cl_corr > 1.6:
        fitness -= 20 * (cl_corr - 1.6)

    smooth_penalty = np.sum(np.maximum(0, np.abs(np.diff(delta_y_smooth)) - 0.05) ** 2)
    fitness -= 0.2 * smooth_penalty / num_ctrl

    d2yc_dx2 = np.gradient(np.gradient(yc, x_dense), x_dense)
    center_mask_penalty = (x_dense > 0.40) & (x_dense < 0.60)
    curvature_violation = np.maximum(0, np.abs(d2yc_dx2[center_mask_penalty]) - 0.8)
    fitness -= 1.5 * np.mean(curvature_violation ** 2)

    fitness -= 0.1 * (np.std(delta_y_smooth[:5]) + np.std(delta_y_smooth[5:])) / 0.05

    d2yu_dx2 = np.gradient(np.gradient(yu, x_dense), x_dense)
    d2yl_dx2 = np.gradient(np.gradient(yl, x_dense), x_dense)
    mask = (x_dense > 0.2) & (x_dense < 0.8)
    fitness -= 0.8 * (np.mean(d2yu_dx2[mask] ** 2) + np.mean(d2yl_dx2[mask] ** 2))

    fitness += 0.05 * np.sum(np.abs(delta_y_smooth))

    genome.cm = cm_corr
    genome.clcd = cl_corr / cd_corr

    return fitness

# ================== NEAT TRAINING ===========================
def train_for_aoa(target_aoa, generations=50, seed_genome=None):
    global CURRENT_GEN
    overall_best_genome = None
    overall_best_fitness = -np.inf

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "NEAT Config Single Genome.ini"
    )

    # Initialize population
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    if seed_genome is not None:
        genome0 = list(pop.population.values())[0]

        # Copy weights ONLY for matching connections
        for key in genome0.connections:
            if key in seed_genome.connections:
                genome0.nodes[key].bias = seed_genome.nodes[key].bias

        print("Seed genome weights copied (structure preserved)")

    # ===== LIVE TRAINING PLOT AS SUBPLOTS =====
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"NEAT Training Convergence — AoA {target_aoa}°, Re={re_folder}",
        fontsize=16,
        fontweight='bold'
    )

    cm_history, clcd_history = [], []

    # Plot CM and CM limit on the same axis
    cm_line, = ax1.plot([], [], color='#1f77b4', label="|CM|")
    limit_line, = ax1.plot([], [], "--", color='gray', label="CM Limit")
    ax1.set_ylabel("Pitching Moment", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_title("Live Pitching Moment (CM) during NEAT Training", fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)

    # CL/CD subplot
    ax2_line, = ax2.plot([], [], color='#ff7f0e', label="CL/CD")
    ax2.set_ylabel("Corrected CL/CD", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title("Live Corrected CL/CD during NEAT Training", fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)

    def eval_genomes(genomes, config):
        nonlocal overall_best_genome, overall_best_fitness
        global CURRENT_GEN

        gen_best_genome = None
        gen_best_fitness = -np.inf

        # Evaluate all genomes in the generation
        for gid, genome in genomes:
            genome.fitness = compute_fitness(genome, config, target_aoa)

            if genome.fitness > gen_best_fitness:
                gen_best_fitness = genome.fitness
                gen_best_genome = genome

        # Update overall best genome if needed
        if gen_best_fitness > overall_best_fitness:
            overall_best_fitness = gen_best_fitness
            overall_best_genome = gen_best_genome

        # Use the overall best genome for live plotting
        best_cm = abs(overall_best_genome.cm)
        best_ld = overall_best_genome.clcd
        cm_history.append(best_cm)
        clcd_history.append(best_ld)

        # Update live plots
        cm_line.set_data(range(len(cm_history)), cm_history)
        limit_line.set_data(range(len(cm_history)), [get_cm_limit(g) for g in range(len(cm_history))])
        ax2_line.set_data(range(len(clcd_history)), clcd_history)
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim();
        ax2.autoscale_view()
        plt.draw();
        plt.pause(0.001)

        CURRENT_GEN += 1

    winner = pop.run(eval_genomes, generations)

    # ================== SAVE BEST GENOME ==================
    aoa_folder = f"{target_aoa:.2f} Degrees"
    save_dir = os.path.join("BestGenomes", f"Re{re_folder}", aoa_folder)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "best_genome_nf.pkl"), "wb") as f:
        pickle.dump(winner, f)
    draw_net(config, winner, view=False, filename=os.path.join(save_dir, "best_network"))

    # ================== FINAL CONVERGENCE PLOTS ==================
    plt.ioff()  # turn off interactive mode

    fig_final, (ax1_final, ax2_final) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig_final.suptitle(
        f"NEAT Training Convergence — AoA {target_aoa}°, Re={re_folder}",
        fontsize=16,
        fontweight='bold'
    )

    # ---------- CM subplot ----------
    ax1_final.plot(cm_history, color='#1f77b4', label="|CM|")  # blue
    ax1_final.plot([get_cm_limit(g) for g in range(len(cm_history))], '--', color='gray', label="CM Limit")  # CM limit

    ax1_final.set_ylabel("|CM|", fontsize=12, fontweight='bold')
    ax1_final.set_title("Pitching Moment (CM) Convergence", fontsize=14, fontweight='bold')
    ax1_final.grid(True, linestyle='--', alpha=0.5)
    ax1_final.legend(loc='upper right', fontsize=10)

    # ---------- CL/CD subplot ----------
    ax2_final.plot(clcd_history, color='#ff7f0e', label="CL/CD")  # orange

    ax2_final.set_ylabel("Corrected CL/CD", fontsize=12, fontweight='bold')
    ax2_final.set_xlabel("Generation", fontsize=12, fontweight='bold')
    ax2_final.set_title("Lift-to-Drag Ratio (CL/CD) Convergence", fontsize=14, fontweight='bold')
    ax2_final.grid(True, linestyle='--', alpha=0.5)
    ax2_final.legend(loc='upper right', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ================= SAVE FIGURE =================
    plot_path = os.path.join(save_dir, "training_convergence.png")
    fig_final.savefig(plot_path, dpi=300)

    print(f"Saved convergence plot to: {plot_path}")

    # show the final plot briefly
    plt.show(block=False)
    plt.pause(3)  # seconds to display the plot

    # ================= CLEANUP =================
    plt.close(fig)
    plt.close(fig_final)
    plt.close('all')

# ================== RUN TRAINING ===========================
train_for_aoa(-3.75, generations=Gen)

# aoa_values = np.arange(-5.00, 12.50 + 0.001, 0.25)
#
# seed_genome = None
# for i, aoa in enumerate(aoa_values):
#     print("\n=======================================")
#     print(f"Starting training for AoA = {aoa:.2f}°")
#     print("=======================================\n")
#
#     CURRENT_GEN = 0
#
#     # Load previous AoA best genome as seed
#     if i > 0:
#         prev_aoa = aoa_values[i-1]
#         prev_folder = os.path.join("BestGenomes", f"Re{re_folder}", f"{prev_aoa:.2f} Degrees")
#         prev_file = os.path.join(prev_folder, "best_genome_nf.pkl")
#         if os.path.exists(prev_file):
#             with open(prev_file, "rb") as f:
#                 seed_genome = pickle.load(f)
#             print(f"Loaded best genome from AoA {prev_aoa:.2f}° as seed.")
#         else:
#             seed_genome = None
#             print("No previous genome found; starting from scratch.")
#
#     train_for_aoa(aoa, generations=Gen, seed_genome=seed_genome)