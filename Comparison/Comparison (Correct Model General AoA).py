import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import joblib

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

geom_dir = "../Morphing/Geometry/"  # folder with saved .txt airfoils
xfoil_dir = "../XFOIL/Simulation Results/"
nf_dir = "../NeuralFoil/Simulation Results/"
comparison_dir = "../Comparison/Comparison Results"
Re = 1e6

os.makedirs(comparison_dir, exist_ok=True)
model_dir = os.path.join(comparison_dir, "global_model")  # <--- UPDATED


# ============================================================
# === FILE READING FUNCTIONS =================================
# ============================================================

def read_geometry_file(filename):
    """Read airfoil geometry .txt file and extract control points (x_ctrl, y_ctrl)."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        idx_ctrl = [i for i, l in enumerate(lines) if '=== Control Points' in l][0] + 1
        ctrl_lines = []
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
    """Reads XFOIL or NeuralFoil polar file and extracts alpha, Cl, Cd, and Cm."""
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


# ============================================================
# === LOAD GLOBAL MODELS =====================================
# ============================================================

model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb.joblib"))

# ============================================================
# === SELECT AIRFOIL =========================================
# ============================================================

airfoil_number = "380"
geom_file = os.path.join(geom_dir, f"airfoil_points_{airfoil_number}.txt")
file_xfoil = os.path.join(xfoil_dir, f"polar_XFOIL_{airfoil_number}_Re{int(Re):.0f}.txt")
file_nf = os.path.join(nf_dir, f"polar_NeuralFoil_{airfoil_number}_Re{int(Re):.0f}.txt")

# Load base geometry (for dy_vec)
geom_files = sorted(glob.glob(os.path.join(geom_dir, "airfoil_points_*.txt")))
x_base, y_base = read_geometry_file(geom_files[0])
x_ctrl, y_ctrl = read_geometry_file(geom_file)
dy_vec = (y_ctrl - y_base).astype(float)

# === REBUILD TRAINING FEATURES (must match training code) ===
dy_cumsum = np.cumsum(dy_vec)
dy_dx = np.gradient(dy_vec, x_ctrl)
d2y_dx2 = np.gradient(dy_dx, x_ctrl)

base_features = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2])

# Load polars
alpha_x, cl_x, cd_x, cm_x, _ = read_polar(file_xfoil)
alpha_nf, cl_nf, cd_nf, cm_nf, _ = read_polar(file_nf)

# Interpolate NF to XFOIL AoA grid
f_cl = interp1d(alpha_nf, cl_nf, fill_value="extrapolate")
f_cd = interp1d(alpha_nf, cd_nf, fill_value="extrapolate")
f_cm = interp1d(alpha_nf, cm_nf, fill_value="extrapolate")
cl_nf_i = f_cl(alpha_x)
cd_nf_i = f_cd(alpha_x)
cm_nf_i = f_cm(alpha_x)


# ============================================================
# === APPLY GLOBAL CORRECTION ================================
# ============================================================

cl_corr = []
cd_corr = []
cm_corr = []

for idx, a in enumerate(alpha_x):
    X_input = np.hstack([base_features, a]).reshape(1, -1)

    err_cl = model_cl.predict(X_input)[0]
    err_cd = model_cd.predict(X_input)[0]
    err_cm = model_cm.predict(X_input)[0]

    # Apply correction: NF - predicted error
    cl_corr.append(cl_nf_i[idx] - err_cl)
    cd_corr.append(cd_nf_i[idx] - err_cd)
    cm_corr.append(cm_nf_i[idx] - err_cm)

cl_corr = np.array(cl_corr)
cd_corr = np.array(cd_corr)
cm_corr = np.array(cm_corr)


# ============================================================
# === COMPUTE ERRORS =========================================
# ============================================================

err_before = cl_nf_i - cl_x
err_after = cl_corr - cl_x

mean_abs_cl_before = np.mean(np.abs(err_before))
mean_abs_cl_after = np.mean(np.abs(err_after))

print(f"Mean |ΔCl| before correction: {mean_abs_cl_before:.4f}")
print(f"Mean |ΔCl| after correction : {mean_abs_cl_after:.4f}")


# ============================================================
# === SAVE DETAILED CSV ======================================
# ============================================================

output_csv = os.path.join(comparison_dir,
                          f"comparison_corrected_{airfoil_number}_Re{int(Re):.0f}.csv")

df = pd.DataFrame({
    "AoA": alpha_x,
    "Cl_XFOIL": cl_x,
    "Cl_NF": cl_nf_i,
    "Cl_Corr": cl_corr,
    "Cd_XFOIL": cd_x,
    "Cd_NF": cd_nf_i,
    "Cd_Corr": cd_corr,
    "Cm_XFOIL": cm_x,
    "Cm_NF": cm_nf_i,
    "Cm_Corr": cm_corr,
    "err_Cl_before": err_before,
    "err_Cl_after": err_after
})
df.to_csv(output_csv, index=False)
print(f"💾 Corrected comparison saved to '{output_csv}'")


# ============================================================
# === PLOTTING ===============================================
# ============================================================

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(3, 2, figsize=(13, 10))
fig.suptitle(f"XFOIL vs NeuralFoil Comparison (Corrected) — Airfoil {airfoil_number} (Re={int(Re):.0f})",
             fontsize=14, weight='bold')

# CL
axs[0, 0].plot(alpha_x, cl_x, 'o-', label='XFOIL')
axs[0, 0].plot(alpha_x, cl_nf_i, 's--', label='NF')
axs[0, 0].plot(alpha_x, cl_corr, 'd-', label='NF Corr')
axs[0, 0].set_ylabel("Cl")
axs[0, 0].legend()
axs[0, 0].set_title("Lift Coefficient")

axs[0, 1].plot(alpha_x, err_before, 'r-', label='Before Corr')
axs[0, 1].plot(alpha_x, err_after, 'g-', label='After Corr')
axs[0, 1].axhline(0, color='k', lw=0.8)
axs[0, 1].set_ylabel("ΔCl")
axs[0, 1].set_title("Cl Error vs AoA")
axs[0, 1].legend()

# CD
axs[1, 0].plot(alpha_x, cd_x, 'o-', label='XFOIL')
axs[1, 0].plot(alpha_x, cd_nf_i, 's--', label='NF')
axs[1, 0].plot(alpha_x, cd_corr, 'd-', label='NF Corr')
axs[1, 0].set_ylabel("Cd")
axs[1, 0].legend()
axs[1, 0].set_title("Drag Coefficient")

axs[1, 1].plot(alpha_x, cd_nf_i - cd_x, 'r-', label='Before Corr')
axs[1, 1].plot(alpha_x, cd_corr - cd_x, 'g-', label='After Corr')
axs[1, 1].axhline(0, color='k', lw=0.8)
axs[1, 1].set_ylabel("ΔCd")
axs[1, 1].set_title("Cd Error vs AoA")
axs[1, 1].legend()

# CM
axs[2, 0].plot(alpha_x, cm_x, 'o-', label='XFOIL')
axs[2, 0].plot(alpha_x, cm_nf_i, 's--', label='NF')
axs[2, 0].plot(alpha_x, cm_corr, 'd-', label='NF Corr')
axs[2, 0].set_ylabel("Cm")
axs[2, 0].legend()
axs[2, 0].set_title("Pitching Moment Coefficient")
axs[2, 0].set_xlabel("AoA (deg)")

axs[2, 1].plot(alpha_x, cm_nf_i - cm_x, 'r-', label='Before Corr')
axs[2, 1].plot(alpha_x, cm_corr - cm_x, 'g-', label='After Corr')
axs[2, 1].axhline(0, color='k', lw=0.8)
axs[2, 1].set_ylabel("ΔCm")
axs[2, 1].set_title("Cm Error vs AoA")
axs[2, 1].set_xlabel("AoA (deg)")
axs[2, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()