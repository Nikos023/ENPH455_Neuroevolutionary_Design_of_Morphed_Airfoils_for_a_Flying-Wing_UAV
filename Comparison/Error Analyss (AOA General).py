import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

geom_dir = "../Morphing/Geometry/"
comparison_dir = "../Comparison/Comparison Results"

Re = 3e5

def format_re(re):
    return f"{re:.0e}".replace("+0", "").replace("+", "")

re_folder = f"Re{format_re(Re)}"

xfoil_dir = os.path.join("../XFOIL", f"Simulation Results 5000{re_folder}")
nf_dir    = os.path.join("../NeuralFoil", f"Simulation Results 5000{re_folder}")

model_dir = os.path.join(
    comparison_dir,
    "global_model",
    "5000gb",
    re_folder
)

os.makedirs(model_dir, exist_ok=True)

# ============================================================
# === UTIL: read geometry & polar & compute errors ===========
# ============================================================

def read_geometry_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    try:
        idx_ctrl = [i for i, l in enumerate(lines)
                    if '=== Control Points' in l][0] + 1

        ctrl_lines = []
        for line in lines[idx_ctrl:]:
            if '===' in line:
                break
            line = line.strip()
            if line and not line.startswith("#"):
                ctrl_lines.append(line)

        ctrl_points = np.array(
            [[float(val) for val in l.split(',')] for l in ctrl_lines]
        )
        x_ctrl = ctrl_points[:, 0]
        y_ctrl = ctrl_points[:, 1]

    except Exception as e:
        print(f"⚠️ Control point parsing failed for {filename}: {e}")
        x_ctrl, y_ctrl = np.array([]), np.array([])

    return x_ctrl, y_ctrl


def read_polar(filename):
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


def compute_errors(alpha_x, cl_x, cd_x, cm_x,
                   alpha_nf, cl_nf, cd_nf, cm_nf):

    f_cl = interp1d(alpha_nf, cl_nf, fill_value="extrapolate")
    f_cd = interp1d(alpha_nf, cd_nf, fill_value="extrapolate")
    f_cm = interp1d(alpha_nf, cm_nf, fill_value="extrapolate")

    cl_nf_i = f_cl(alpha_x)
    cd_nf_i = f_cd(alpha_x)
    cm_nf_i = f_cm(alpha_x)

    return cl_nf_i - cl_x, cd_nf_i - cd_x, cm_nf_i - cm_x


# ============================================================
# === COLLECT DATA ===========================================
# ============================================================

geom_files = sorted(glob.glob(os.path.join(geom_dir, "airfoil_points_*.txt")))
if not geom_files:
    raise FileNotFoundError(f"No geometry .txt files found in {geom_dir}")

x_base, y_base = read_geometry_file(geom_files[0])

global_X = []
global_ecl = []
global_ecd = []
global_ecm = []

n_skipped = 0

for geom_file in geom_files:

    airfoil_name = os.path.splitext(os.path.basename(geom_file))[0]
    airfoil_number = airfoil_name.split("_")[-1]

    x_ctrl, y_ctrl = read_geometry_file(geom_file)

    if len(y_ctrl) == 0 or len(y_ctrl) != len(y_base):
        print(f"Skipping {airfoil_name}: control points missing or mismatch")
        n_skipped += 1
        continue

    dy_vec = (y_ctrl - y_base).astype(float)

    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)

    base_features = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2])

    file_xfoil = os.path.join(
        xfoil_dir,
        f"polar_XFOIL_{airfoil_number}_Re{int(Re)}.txt"
    )

    file_nf = os.path.join(
        nf_dir,
        f"polar_NeuralFoil_{airfoil_number}_Re{int(Re)}.txt"
    )

    if not (os.path.isfile(file_xfoil) and os.path.isfile(file_nf)):
        print(f"Skipping {airfoil_name}: missing polar files")
        n_skipped += 1
        continue

    alpha_x, cl_x, cd_x, cm_x, _ = read_polar(file_xfoil)
    alpha_nf, cl_nf, cd_nf, cm_nf, _ = read_polar(file_nf)

    err_cl, err_cd, err_cm = compute_errors(
        alpha_x, cl_x, cd_x, cm_x,
        alpha_nf, cl_nf, cd_nf, cm_nf
    )

    mask = (
        (np.abs(err_cl) < 0.5) &
        (np.abs(err_cd) < 0.05) &
        (np.abs(err_cm) < 0.05)
    )

    alpha_x = alpha_x[mask]
    err_cl = err_cl[mask]
    err_cd = err_cd[mask]
    err_cm = err_cm[mask]

    for a, ecl, ecd, ecm in zip(alpha_x, err_cl, err_cd, err_cm):
        global_X.append(np.hstack([base_features, a]))
        global_ecl.append(ecl)
        global_ecd.append(ecd)
        global_ecm.append(ecm)

print(f"\nFiles processed: {len(geom_files)}, skipped: {n_skipped}")
print(f"Total samples: {len(global_X)}")


# ============================================================
# === TRAIN GLOBAL MODEL =====================================
# ============================================================

from sklearn.ensemble import GradientBoostingRegressor
import joblib

X = np.vstack(global_X)
y_cl = np.array(global_ecl)
y_cd = np.array(global_ecd)
y_cm = np.array(global_ecm)

if X.shape[1] != 41:
    raise ValueError("❌ Feature size mismatch — expected 41 features.")

def build_gb():
    return GradientBoostingRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.9,
        min_samples_leaf=3,
        verbose=1
    )

print("\nTraining Cl model...")
model_cl = build_gb()
model_cl.fit(X, y_cl)

print("\nTraining Cd model...")
model_cd = build_gb()
model_cd.fit(X, y_cd)

print("\nTraining Cm model...")
model_cm = build_gb()
model_cm.fit(X, y_cm)

joblib.dump(model_cl, os.path.join(model_dir, "global_cl_gb.joblib"))
joblib.dump(model_cd, os.path.join(model_dir, "global_cd_gb.joblib"))
joblib.dump(model_cm, os.path.join(model_dir, "global_cm_gb.joblib"))

print("\n✅ Models saved to:", model_dir)


# ============================================================
# === SAVE MODEL SUMMARY =====================================
# ============================================================

pd.DataFrame([{
    "n_samples": len(X),
    "Re": re_folder,
    "model": "GradientBoosting (41 features)"
}]).to_csv(os.path.join(model_dir, "model_summary.csv"), index=False)

print("📄 Model summary saved.")
print("🎉 Training complete.")