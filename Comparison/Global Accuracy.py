import os
import glob
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import joblib

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

geom_dir = "../Morphing/Geometry/"
comparison_dir = "../Comparison/Comparison Results"

Re_list = [1e6, 5e5, 2e5, 1e5, 5e4]

alpha_common = np.linspace(-5, 12, 200)
EPS = 1e-6
MAX_AIRFOILS = 2001

# ============================================================
# === HELPERS ================================================
# ============================================================

def format_re(re):
    return f"{re:.0e}".replace("+0", "").replace("+", "")

def mae(pred, truth):
    return np.mean(np.abs(pred - truth))

def rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth)**2))

def read_geometry_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    idx_ctrl = [i for i, l in enumerate(lines) if '=== Control Points' in l][0] + 1
    ctrl = []
    for line in lines[idx_ctrl:]:
        if '===' in line:
            break
        if ',' in line:
            ctrl.append([float(v) for v in line.split(',')])
    ctrl = np.array(ctrl)
    return ctrl[:, 0], ctrl[:, 1]

def read_polar(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    if any("neuralfoil" in l.lower() for l in lines):
        skip = 5
        nf = True
    else:
        skip = 12
        nf = False
    data = np.loadtxt(filename, skiprows=skip)
    if nf:
        return data[:,0], data[:,1], data[:,2], data[:,3]
    else:
        return data[:,0], data[:,1], data[:,2], data[:,4]

# ============================================================
# === CORE FUNCTION ==========================================
# ============================================================

def run_for_re(Re):
    print(f"\n{'='*60}")
    print(f"🚀 Running for Re = {Re:.0e}".replace("+0","").replace("+",""))
    print(f"{'='*60}")

    re_folder = f"Re{format_re(Re)}"
    xfoil_dir = os.path.join("../XFOIL", f"Simulation Results 5000{re_folder}")
    nf_dir    = os.path.join("../NeuralFoil", f"Simulation Results 5000{re_folder}")
    model_dir = os.path.join(comparison_dir, "global_model/2000gb", re_folder)

    # Load ML models
    model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb.joblib"))
    model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb.joblib"))
    model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb.joblib"))

    errors_nf = {"Cl": [], "Cd": [], "Cm": []}
    errors_corr = {"Cl": [], "Cd": [], "Cm": []}

    geom_files = sorted(glob.glob(os.path.join(geom_dir, "airfoil_points_*.txt")))
    geom_files = geom_files[:MAX_AIRFOILS]
    x_base, y_base = read_geometry_file(geom_files[0])

    for geom_file in geom_files:
        airfoil_id = geom_file.split("_")[-1].split(".")[0]
        file_xfoil = os.path.join(xfoil_dir, f"polar_XFOIL_{airfoil_id}_Re{int(Re)}.txt")
        file_nf = os.path.join(nf_dir, f"polar_NeuralFoil_{airfoil_id}_Re{int(Re)}.txt")

        if not (os.path.exists(file_xfoil) and os.path.exists(file_nf)):
            continue

        # --- Geometry Features ---
        x_ctrl, y_ctrl = read_geometry_file(geom_file)
        dy = y_ctrl - y_base
        base_features = np.hstack([
            dy,
            np.cumsum(dy),
            np.gradient(dy, x_ctrl),
            np.gradient(np.gradient(dy, x_ctrl), x_ctrl)
        ])

        # --- Polars ---
        alpha_x, cl_x, cd_x, cm_x = read_polar(file_xfoil)
        alpha_nf, cl_nf, cd_nf, cm_nf = read_polar(file_nf)

        cl_nf_i = interp1d(alpha_nf, cl_nf, fill_value="extrapolate")(alpha_x)
        cd_nf_i = interp1d(alpha_nf, cd_nf, fill_value="extrapolate")(alpha_x)
        cm_nf_i = interp1d(alpha_nf, cm_nf, fill_value="extrapolate")(alpha_x)

        cl_corr, cd_corr, cm_corr = [], [], []

        for i, a in enumerate(alpha_x):
            X = np.hstack([base_features, a]).reshape(1, -1)
            cl_corr.append(cl_nf_i[i] - model_cl.predict(X)[0])
            cd_corr.append(cd_nf_i[i] - model_cd.predict(X)[0])
            cm_corr.append(cm_nf_i[i] - model_cm.predict(X)[0])

        cl_corr = np.array(cl_corr)
        cd_corr = np.array(cd_corr)
        cm_corr = np.array(cm_corr)

        # --- Errors ---
        err_nf = {
            "Cl": np.interp(alpha_common, alpha_x, np.abs(cl_nf_i - cl_x)),
            "Cd": np.interp(alpha_common, alpha_x, np.abs(cd_nf_i - cd_x)),
            "Cm": np.interp(alpha_common, alpha_x, np.abs(cm_nf_i - cm_x)),
        }
        err_corr = {
            "Cl": np.interp(alpha_common, alpha_x, np.abs(cl_corr - cl_x)),
            "Cd": np.interp(alpha_common, alpha_x, np.abs(cd_corr - cd_x)),
            "Cm": np.interp(alpha_common, alpha_x, np.abs(cm_corr - cm_x)),
        }

        for c in ["Cl","Cd","Cm"]:
            errors_nf[c].append(err_nf[c])
            errors_corr[c].append(err_corr[c])

    # --- Plot Median ± IQR ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    for ax, coeff, title in zip(axs, ["Cl", "Cd", "Cm"], ["Lift", "Drag", "Moment"]):
        nf = np.array(errors_nf[coeff])
        corr = np.array(errors_corr[coeff])
        median_nf = np.median(nf, axis=0)
        median_corr = np.median(corr, axis=0)
        iqr_nf = np.percentile(nf, [25, 75], axis=0)
        iqr_corr = np.percentile(corr, [25, 75], axis=0)

        ax.plot(alpha_common, median_nf, label="NeuralFoil")
        ax.fill_between(alpha_common, iqr_nf[0], iqr_nf[1], alpha=0.25)
        ax.plot(alpha_common, median_corr, label="Corrected NeuralFoil")
        ax.fill_between(alpha_common, iqr_corr[0], iqr_corr[1], alpha=0.25)

        ax.set_title(f"{title} Absolute Error vs XFOIL Results", fontsize=12)
        ax.set_ylabel(f"{title} |Error|", fontsize=12)

    axs[-1].set_xlabel("Angle of Attack (°)", fontsize=12)
    axs[0].legend()
    plt.suptitle(
        f"Global NeuralFoil Performance vs XFOIL Results at Re = {Re:.0e}".replace("+0", "").replace("+",
                                                                                                     "") + "\nMedian ± IQR",
        fontsize=14, weight="bold"
    )
    plt.show()

    # --- Return Error Reduction ---
    improv = {}
    for coeff in ["Cl","Cd","Cm"]:
        nf_c = np.concatenate(errors_nf[coeff])
        corr_c = np.concatenate(errors_corr[coeff])
        mae_nf_c = mae(nf_c, 0)
        mae_corr_c = mae(corr_c, 0)
        improv[coeff] = 100 * (mae_nf_c - mae_corr_c) / (mae_nf_c + EPS)

    # --- Compute combined improvement ---
    combined_nf = np.concatenate([np.concatenate(errors_nf["Cl"]),
                                  np.concatenate(errors_nf["Cd"]),
                                  np.concatenate(errors_nf["Cm"])])
    combined_corr = np.concatenate([np.concatenate(errors_corr["Cl"]),
                                    np.concatenate(errors_corr["Cd"]),
                                    np.concatenate(errors_corr["Cm"])])
    mae_nf_all = mae(combined_nf, 0)
    mae_corr_all = mae(combined_corr, 0)
    improv["All"] = 100 * (mae_nf_all - mae_corr_all) / (mae_nf_all + EPS)

    # --- Print improvements ---
    print(f"✅ Improvements for Re = {Re:.0e}:")
    print(f"    Cl Error Reduction      : {improv['Cl']:.2f}%")
    print(f"    Cd Error Reduction      : {improv['Cd']:.2f}%")
    print(f"    Cm Error Reduction      : {improv['Cm']:.2f}%")
    print(f"    Combined All Error Red. : {improv['All']:.2f}%")

    return improv

# ============================================================
# === MAIN LOOP ==============================================
# ============================================================

re_labels = []
improvements_cl = []
improvements_cd = []
improvements_cm = []
improvements_all = []

for Re in Re_list:
    result = run_for_re(Re)
    re_labels.append(Re)
    improvements_cl.append(result["Cl"])
    improvements_cd.append(result["Cd"])
    improvements_cm.append(result["Cm"])
    improvements_all.append(result["All"])

# ============================================================
# === ERROR REDUCTION VS Re =================================
# ============================================================

import matplotlib.pyplot as plt

# Reynolds numbers
Re_list = [1e6, 5e5, 2e5, 1e5, 5e4]

# Error reduction (%) from your printed results
improvements_cl  = [68.99, 62.53, 59.62, 61.94, 45.94]
improvements_cd  = [71.00, 66.61, 64.43, 70.30, 55.82]
improvements_cm  = [79.98, 74.90, 65.91, 61.64, 50.83]
improvements_all = [71.60, 65.46, 60.97, 62.63, 47.34]

# Plot
plt.figure(figsize=(10,6))
plt.plot(Re_list, improvements_cl, 'o-', label="Cl")
plt.plot(Re_list, improvements_cd, 's-', label="Cd")
plt.plot(Re_list, improvements_cm, '^-', label="Cm")
plt.plot(Re_list, improvements_all, 'd-', label="All Combined")

plt.xscale("log")
plt.xlabel("Reynolds Number", fontweight='bold')
plt.ylabel("Error Reduction (%)", fontweight='bold')
plt.title("Global ML Correction Error Reduction vs Reynolds Number", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()