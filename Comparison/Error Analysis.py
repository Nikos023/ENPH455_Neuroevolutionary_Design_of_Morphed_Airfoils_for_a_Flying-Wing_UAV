import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import glob
import pandas as pd

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

geom_dir = "../Morphing/Geometry/"  # folder with saved .txt airfoils
xfoil_dir = "../XFOIL/Simulation Results/"
nf_dir = "../NeuralFoil/Simulation Results/"
comparison_dir = "../Comparison/Comparison Results"
Re = 1e6

os.makedirs(comparison_dir, exist_ok=True)

# ============================================================
# === FUNCTIONS ==============================================
# ============================================================

def read_geometry_file(filename):
    """Read airfoil geometry .txt file and extract control points."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        # Find the "=== Control Points ===" section
        idx_ctrl = [i for i, l in enumerate(lines) if '=== Control Points' in l][0] + 1
        ctrl_lines = []

        # Read until the next "===" line OR end of file
        for line in lines[idx_ctrl:]:
            if '===' in line:
                break
            line = line.strip()
            if line and not line.startswith("#"):
                ctrl_lines.append(line)

        ctrl_points = np.array([[float(val) for val in l.split(',')] for l in ctrl_lines])
        x_ctrl = ctrl_points[:, 0]
        y_ctrl = ctrl_points[:, 1]
    except Exception as e:
        print(f"⚠️ Control point parsing failed for {filename}: {e}")
        x_ctrl, y_ctrl = np.array([]), np.array([])

    return x_ctrl, y_ctrl


def read_polar(filename):
    """Read polar file and extract alpha, Cl, Cd, Cm."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    if any("neuralfoil" in line.lower() for line in lines):
        filetype = "neuralfoil"
        skiprows = 5
    elif any("xfoil" in line.lower() for line in lines):
        filetype = "xfoil"
        skiprows = 12
    else:
        raise ValueError(f"Unknown file format for {filename}")

    data = np.loadtxt(filename, skiprows=skiprows)
    if filetype == "xfoil":
        alpha, cl, cd, cm = data[:, 0], data[:, 1], data[:, 2], data[:, 4]
    else:
        alpha, cl, cd, cm = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return alpha, cl, cd, cm, filetype


def compute_errors(alpha_x, cl_x, cd_x, cm_x, alpha_nf, cl_nf, cd_nf, cm_nf):
    """Interpolate NeuralFoil data to XFOIL AoAs and compute absolute errors."""
    f_cl = interp1d(alpha_nf, cl_nf, fill_value="extrapolate")
    f_cd = interp1d(alpha_nf, cd_nf, fill_value="extrapolate")
    f_cm = interp1d(alpha_nf, cm_nf, fill_value="extrapolate")

    cl_nf_i = f_cl(alpha_x)
    cd_nf_i = f_cd(alpha_x)
    cm_nf_i = f_cm(alpha_x)

    err_cl = cl_nf_i - cl_x
    err_cd = cd_nf_i - cd_x
    err_cm = cm_nf_i - cm_x

    # No scaling, just return absolute errors
    return err_cl, err_cd, err_cm


# ============================================================
# === MAIN LOOP ==============================================
# ============================================================

summary_list = []
disp_matrix = []
rms_Cl, rms_Cd, rms_Cm = [], [], []

geom_files = sorted(glob.glob(os.path.join(geom_dir, "airfoil_points_*.txt")))
if not geom_files:
    raise FileNotFoundError(f"No geometry .txt files found in {geom_dir}")

# Baseline geometry
baseline_geom = geom_files[0]
x_base, y_base = read_geometry_file(baseline_geom)

for geom_file in geom_files:
    airfoil_base = os.path.splitext(os.path.basename(geom_file))[0]
    airfoil_number = airfoil_base.split("_")[-1]

    x_ctrl, y_ctrl = read_geometry_file(geom_file)
    if len(y_ctrl) == 0:
        print(f"Skipping {airfoil_base}: no control points found")
        continue

    # === Polar files ===
    file_xfoil = os.path.join(xfoil_dir, f"polar_XFOIL_{airfoil_number}_Re{int(Re):.0f}.txt")
    file_nf = os.path.join(nf_dir, f"polar_NeuralFoil_{airfoil_number}_Re{int(Re):.0f}.txt")

    if not os.path.isfile(file_xfoil) or not os.path.isfile(file_nf):
        print(f"Skipping {geom_file}: missing polar files")
        continue

    alpha_x, cl_x, cd_x, cm_x, _ = read_polar(file_xfoil)
    alpha_nf, cl_nf, cd_nf, cm_nf, _ = read_polar(file_nf)

    err_cl, err_cd, err_cm = compute_errors(
        alpha_x, cl_x, cd_x, cm_x, alpha_nf, cl_nf, cd_nf, cm_nf)

    # RMS absolute errors
    rms_pct_cl = np.sqrt(np.mean(err_cl**2))
    rms_pct_cd = np.sqrt(np.mean(err_cd**2))
    rms_pct_cm = np.sqrt(np.mean(err_cm**2))

    summary_list.append({
        "Airfoil": airfoil_base,
        "RMS_Cl": rms_pct_cl,
        "RMS_Cd": rms_pct_cd,
        "RMS_Cm": rms_pct_cm
    })

    if len(y_ctrl) == len(y_base):
        disp_matrix.append(y_ctrl - y_base)
        rms_Cl.append(rms_pct_cl)
        rms_Cd.append(rms_pct_cd)
        rms_Cm.append(rms_pct_cm)

# ============================================================
# === SAVE SUMMARY & PLOT ===================================
# ============================================================

if summary_list:
    df_summary = pd.DataFrame(summary_list)
    csv_file = os.path.join(comparison_dir, f"NeuralFoil_vs_XFOIL_Re{int(Re):.0f}.csv")
    df_summary.to_csv(csv_file, index=False)
    print(f"✅ Master summary saved to {csv_file}")

    disp_matrix = np.array(disp_matrix)
    rms_Cl = np.array(rms_Cl)
    rms_Cd = np.array(rms_Cd)
    rms_Cm = np.array(rms_Cm)

    n_ctrl = disp_matrix.shape[1]
    ncols = 5
    nrows = int(np.ceil(n_ctrl / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 8), sharey=True)
    axs = axs.ravel()

    for i in range(n_ctrl):
        ax = axs[i]
        x = disp_matrix[:, i]

        for y_data, color, name in zip([rms_Cl, rms_Cd, rms_Cm],
                                       ['tab:blue', 'tab:orange', 'tab:green'],
                                       ['Cl', 'Cd', 'Cm']):
            # --- Linear fit ---
            coef_lin = np.polyfit(x, y_data, 1)
            y_lin_fit = np.polyval(coef_lin, x)

            # --- Quadratic fit ---
            coef_quad = np.polyfit(x, y_data, 2)
            y_quad_fit = np.polyval(coef_quad, x)

            # --- Compute R^2 manually ---
            ss_res_lin = np.sum((y_data - y_lin_fit) ** 2)
            ss_res_quad = np.sum((y_data - y_quad_fit) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

            if ss_tot < 1e-12:
                r2_lin = r2_quad = 1.0
            else:
                r2_lin = 1 - ss_res_lin / ss_tot
                r2_quad = 1 - ss_res_quad / ss_tot

            # --- Choose best fit ---
            if r2_quad > r2_lin:
                best_coef = coef_quad
                best_fit = y_quad_fit
                eqn = f"{name} = {best_coef[0]:.5f}*x² + {best_coef[1]:.5f}*x + {best_coef[2]:.5f}"
            else:
                best_coef = coef_lin
                best_fit = y_lin_fit
                eqn = f"{name} = {best_coef[0]:.5f}*x + {best_coef[1]:.5f}"

            # --- Plot regression ---
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit_plot = np.polyval(best_coef, x_fit)
            ax.plot(x_fit, y_fit_plot, color=color, linestyle='--', linewidth=1)

            # --- Scatter plot ---
            ax.scatter(x, y_data, color=color, alpha=0.7, label=name)

            print(f"Control Point {i + 1}, {eqn}, R² = {max(r2_lin, r2_quad):.4f}")

        ax.axvline(0, color='k', linestyle=':', linewidth=0.8)
        ax.set_title(f"Control Point {i + 1}")
        ax.set_xlabel("Δy (Displacement)")
        if i % ncols == 0:
            ax.set_ylabel("RMS Error")
        ax.grid(True)
        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(n_ctrl, len(axs)):
        axs[j].axis("off")

    plt.suptitle("Regression: Displacement vs RMS Error per Control Point", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

else:
    print("⚠️ No airfoils processed. Check geometry and polar files.")