import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from collections import defaultdict

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

geom_dir = "../Morphing/Geometry/"  # folder with saved .txt airfoils
xfoil_dir = "../XFOIL/Simulation Results/"
nf_dir = "../NeuralFoil/Simulation Results/"
comparison_dir = "../Comparison/Comparison Results"
Re = 1e6

os.makedirs(comparison_dir, exist_ok=True)
model_dir = os.path.join(comparison_dir, "global_model")
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# === UTIL: read geometry & polar & compute errors ===========
# ============================================================

def read_geometry_file(filename):
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
    f_cl = interp1d(alpha_nf, cl_nf, fill_value="extrapolate")
    f_cd = interp1d(alpha_nf, cd_nf, fill_value="extrapolate")
    f_cm = interp1d(alpha_nf, cm_nf, fill_value="extrapolate")
    cl_nf_i = f_cl(alpha_x)
    cd_nf_i = f_cd(alpha_x)
    cm_nf_i = f_cm(alpha_x)
    err_cl = cl_nf_i - cl_x
    err_cd = cd_nf_i - cd_x
    err_cm = cm_nf_i - cm_x
    return err_cl, err_cd, err_cm


# ============================================================
# === COLLECT DATA: CONTROL-POINT DISPLACEMENTS + AoA + ERRORS
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
    x_ctrl, y_ctrl = read_geometry_file(geom_file)

    if len(y_ctrl) == 0 or len(y_ctrl) != len(y_base):
        print(f"Skipping {airfoil_name}: control points missing or mismatch")
        n_skipped += 1
        continue

    dy_vec = (y_ctrl - y_base).astype(float)

    # --- NEW: SEQUENTIAL/CHORDWISE FEATURES ------------------
    # Cumulative sum (leading edge influence downstream)
    dy_cumsum = np.cumsum(dy_vec)
    # First derivative along chord (slope)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    # Second derivative (curvature)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)
    # Combine into feature vector
    base_features = np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2])

    airfoil_number = airfoil_name.split("_")[-1]
    file_xfoil = os.path.join(xfoil_dir, f"polar_XFOIL_{airfoil_number}_Re{int(Re):.0f}.txt")
    file_nf    = os.path.join(nf_dir,    f"polar_NeuralFoil_{airfoil_number}_Re{int(Re):.0f}.txt")

    if not (os.path.isfile(file_xfoil) and os.path.isfile(file_nf)):
        print(f"Skipping {airfoil_name}: missing polar files")
        n_skipped += 1
        continue

    alpha_x, cl_x, cd_x, cm_x, _ = read_polar(file_xfoil)
    alpha_nf, cl_nf, cd_nf, cm_nf, _ = read_polar(file_nf)

    err_cl, err_cd, err_cm = compute_errors(alpha_x, cl_x, cd_x, cm_x,
                                            alpha_nf, cl_nf, cd_nf, cm_nf)

    # --- FILTER OUT POLAR OUTLIERS ----------------------
    mask = (np.abs(err_cl) < 0.5) & (np.abs(err_cd) < 0.05) & (np.abs(err_cm) < 0.05)
    alpha_x = alpha_x[mask]
    err_cl = err_cl[mask]
    err_cd = err_cd[mask]
    err_cm = err_cm[mask]

    # Append features with AoA
    for a, ecl, ecd, ecm in zip(alpha_x, err_cl, err_cd, err_cm):
        global_X.append(np.hstack([base_features, a]))
        global_ecl.append(ecl)
        global_ecd.append(ecd)
        global_ecm.append(ecm)

print(f"\nData collection complete. Files processed: {len(geom_files)}, skipped: {n_skipped}")
print(f"Total samples stored globally: {len(global_X)}")


# ============================================================
# === GLOBAL MODEL TRAINING =================================
# ============================================================

use_sklearn = True
try:
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import SGDRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    import joblib
except Exception:
    use_sklearn = False

X = np.vstack(global_X)
y_cl = np.array(global_ecl)
y_cd = np.array(global_ecd)
y_cm = np.array(global_ecm)

if use_sklearn:

    def build_sgd_pipeline():
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("sgd", SGDRegressor(
                loss="huber",
                penalty="l2",
                alpha=1e-4,
                learning_rate="adaptive",
                eta0=0.01,
                max_iter=50000,
                tol=1e-5,
                early_stopping=True,
                n_iter_no_change=20,
                verbose=1
            ))
        ])

    print("\nTraining global Huber-SGD models...")
    model_cl_sgd = build_sgd_pipeline()
    model_cd_sgd = build_sgd_pipeline()
    model_cm_sgd = build_sgd_pipeline()

    model_cl_sgd.fit(X, y_cl)
    model_cd_sgd.fit(X, y_cd)
    model_cm_sgd.fit(X, y_cm)

    joblib.dump(model_cl_sgd, os.path.join(model_dir, "global_cl_sgd.joblib"))
    joblib.dump(model_cd_sgd, os.path.join(model_dir, "global_cd_sgd.joblib"))
    joblib.dump(model_cm_sgd, os.path.join(model_dir, "global_cm_sgd.joblib"))

    print("\nTraining Gradient Boosting models...")
    model_cl_gb = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        min_samples_leaf=3,
        verbose=1
    )

    model_cd_gb = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        min_samples_leaf=3,
        verbose=1
    )

    model_cm_gb = GradientBoostingRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        min_samples_leaf=3,
        verbose=1
    )

    model_cl_gb.fit(X, y_cl)
    model_cd_gb.fit(X, y_cd)
    model_cm_gb.fit(X, y_cm)

    joblib.dump(model_cl_gb, os.path.join(model_dir, "global_cl_gb.joblib"))
    joblib.dump(model_cd_gb, os.path.join(model_dir, "global_cd_gb.joblib"))
    joblib.dump(model_cm_gb, os.path.join(model_dir, "global_cm_gb.joblib"))

    print("\n✅ All models saved: Huber-SGD and Gradient Boosting")

else:
    X_design = np.hstack([np.ones((len(X), 1)), X])
    coeffs_cl, _, _, _ = np.linalg.lstsq(X_design, y_cl, rcond=None)
    coeffs_cd, _, _, _ = np.linalg.lstsq(X_design, y_cd, rcond=None)
    coeffs_cm, _, _, _ = np.linalg.lstsq(X_design, y_cm, rcond=None)

    np.save(os.path.join(model_dir, "ols_cl.npy"), coeffs_cl)
    np.save(os.path.join(model_dir, "ols_cd.npy"), coeffs_cd)
    np.save(os.path.join(model_dir, "ols_cm.npy"), coeffs_cm)
    print("\n⚠️ Sklearn unavailable — saved OLS instead.")

# ============================================================
# === SAVE MODEL SUMMARY CSV =================================
# ============================================================

pd.DataFrame([{
    "n_samples": len(X),
    "method": "HuberSGD + GradientBoosting" if use_sklearn else "OLS",
    "poly_degree": 4 if use_sklearn else "N/A"
}]).to_csv(os.path.join(model_dir, "model_summary.csv"), index=False)

print("\n📄 Model summary written to model_summary.csv")
print("🎉 Training complete — Global model ready!")