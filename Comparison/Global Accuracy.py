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
xfoil_dir = "../XFOIL/Simulation Results/"
nf_dir = "../NeuralFoil/Simulation Results/"
comparison_dir = "../Comparison/Comparison Results"

Re = 1e6
alpha_common = np.linspace(-5, 12, 200)
EPS = 1e-6

MAX_AIRFOILS = 1000  # <-- maximum number of airfoils to process

os.makedirs(comparison_dir, exist_ok=True)
model_dir = os.path.join(comparison_dir, "global_model")

# ============================================================
# === LOAD GLOBAL MODELS =====================================
# ============================================================

model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb.joblib"))

# ============================================================
# === FILE READING FUNCTIONS =================================
# ============================================================

def read_geometry_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    idx_ctrl = [i for i, l in enumerate(lines) if '=== Control Points' in l][0] + 1
    ctrl = []
    for line in lines[idx_ctrl:]:
        if '===' in line: break
        if ',' in line: ctrl.append([float(v) for v in line.split(',')])
    ctrl = np.array(ctrl)
    return ctrl[:, 0], ctrl[:, 1]

def read_polar(filename):
    with open(filename, 'r') as f: lines = f.readlines()
    if any("neuralfoil" in l.lower() for l in lines):
        skip = 5; nf = True
    else:
        skip = 12; nf = False
    data = np.loadtxt(filename, skiprows=skip)
    if nf: return data[:,0], data[:,1], data[:,2], data[:,3]
    else: return data[:,0], data[:,1], data[:,2], data[:,4]

# ============================================================
# === ERROR METRICS ==========================================
# ============================================================

def mae(pred, truth): return np.mean(np.abs(pred - truth))
def rmse(pred, truth): return np.sqrt(np.mean((pred - truth)**2))
def r2(pred, truth): return 1 - np.sum((pred - np.mean(pred))**2) / (np.sum((truth - np.mean(truth))**2) + EPS)

# ============================================================
# === MAIN ===================================================
# ============================================================

def main():

    errors_nf, errors_corr = {"Cl": [], "Cd": [], "Cm": []}, {"Cl": [], "Cd": [], "Cm": []}

    geom_files = sorted(glob.glob(os.path.join(geom_dir, "airfoil_points_*.txt")))
    geom_files = geom_files[:MAX_AIRFOILS]  # <-- limit number of airfoils
    total_files = len(geom_files)

    if total_files == 0:
        print("⚠️ No geometry files found. Check your paths.")
        return

    x_base, y_base = read_geometry_file(geom_files[0])
    processed, skipped = 0, 0

    print(f"\n🔍 Found {total_files} airfoil geometries (processing up to {MAX_AIRFOILS})\n")

    for idx, geom_file in enumerate(geom_files, start=1):

        airfoil_id = geom_file.split("_")[-1].split(".")[0]
        file_xfoil = os.path.join(xfoil_dir, f"polar_XFOIL_{airfoil_id}_Re{int(Re):.0f}.txt")
        file_nf = os.path.join(nf_dir, f"polar_NeuralFoil_{airfoil_id}_Re{int(Re):.0f}.txt")

        if not (os.path.exists(file_xfoil) and os.path.exists(file_nf)):
            print(f"[{idx}/{total_files}] Airfoil {airfoil_id} → ⚠️ Skipped (missing files)")
            skipped += 1
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

        # --- Corrected Predictions ---
        cl_corr, cd_corr, cm_corr = [], [], []
        for i, a in enumerate(alpha_x):
            X = np.hstack([base_features, a]).reshape(1, -1)
            cl_corr.append(cl_nf_i[i] - model_cl.predict(X)[0])
            cd_corr.append(cd_nf_i[i] - model_cd.predict(X)[0])
            cm_corr.append(cm_nf_i[i] - model_cm.predict(X)[0])
        cl_corr = np.array(cl_corr); cd_corr = np.array(cd_corr); cm_corr = np.array(cm_corr)

        # --- Compute Absolute Errors ---
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

        for coeff in ["Cl","Cd","Cm"]:
            errors_nf[coeff].append(err_nf[coeff])
            errors_corr[coeff].append(err_corr[coeff])

        # --- Progress Print ---
        mae_nf = np.mean([err_nf[c].mean() for c in ["Cl","Cd","Cm"]])
        mae_corr = np.mean([err_corr[c].mean() for c in ["Cl","Cd","Cm"]])
        print(f"[{idx}/{total_files}] Airfoil {airfoil_id} → NF MAE: {mae_nf:.4f}, Corrected MAE: {mae_corr:.4f}")

        processed += 1

    print(f"\n✅ Processing complete: {processed} processed, {skipped} skipped\n")

    # ============================================================
    # === PLOTTING =================================================
    # ============================================================

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3,1, figsize=(12,14), sharex=True)

    for ax, coeff, title in zip(axs, ["Cl","Cd","Cm"], ["Lift","Drag","Moment"]):
        nf = np.array(errors_nf[coeff])
        corr = np.array(errors_corr[coeff])
        median_nf = np.median(nf, axis=0)
        median_corr = np.median(corr, axis=0)
        iqr_nf = np.percentile(nf, [25,75], axis=0)
        iqr_corr = np.percentile(corr, [25,75], axis=0)

        ax.plot(alpha_common, median_nf, label="NeuralFoil", lw=2)
        ax.fill_between(alpha_common, iqr_nf[0], iqr_nf[1], alpha=0.25)
        ax.plot(alpha_common, median_corr, label="Corrected NF", lw=2)
        ax.fill_between(alpha_common, iqr_corr[0], iqr_corr[1], alpha=0.25)

        ax.set_ylabel(f"{title} |Error|")
        ax.set_title(f"{title} Absolute Error vs XFOIL")
        ax.set_ylim(0, max(iqr_nf[1].max(), iqr_corr[1].max())*1.05)

    axs[-1].set_xlabel("Angle of Attack (deg)")
    axs[0].legend()
    plt.suptitle("Global NeuralFoil Performance vs XFOIL\nMedian ± IQR", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # ============================================================
    # === SUMMARY METRICS ========================================
    # ============================================================

    for coeff in ["Cl","Cd","Cm"]:
        nf_all = np.concatenate(errors_nf[coeff])
        corr_all = np.concatenate(errors_corr[coeff])
        print(f"\n{coeff} Metrics:")
        print(f"NeuralFoil:  MAE={mae(nf_all,0):.4f}, RMSE={rmse(nf_all,0):.4f}")
        print(f"Corrected:   MAE={mae(corr_all,0):.4f}, RMSE={rmse(corr_all,0):.4f}")
        print(f"Improvement: {100*(nf_all.mean() - corr_all.mean())/ (nf_all.mean() + 1e-8):.2f}%")

# ============================================================
# === RUN ====================================================
# ============================================================

if __name__=="__main__":
    main()