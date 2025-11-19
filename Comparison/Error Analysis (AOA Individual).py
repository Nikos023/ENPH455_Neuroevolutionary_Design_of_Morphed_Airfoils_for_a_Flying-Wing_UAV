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
model_dir = os.path.join(comparison_dir, "per_aoa_models")
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
# === COLLECT DATA: control-point displacement vectors & errors
# ============================================================

geom_files = sorted(glob.glob(os.path.join(geom_dir, "airfoil_points_*.txt")))
if not geom_files:
    raise FileNotFoundError(f"No geometry .txt files found in {geom_dir}")

x_base, y_base = read_geometry_file(geom_files[0])

aoa_data = defaultdict(lambda: {"dy": [], "ecl": [], "ecd": [], "ecm": [], "airfoil": []})
n_skipped = 0

for geom_file in geom_files:
    airfoil_name = os.path.splitext(os.path.basename(geom_file))[0]
    x_ctrl, y_ctrl = read_geometry_file(geom_file)
    if len(y_ctrl) == 0 or len(y_ctrl) != len(y_base):
        print(f"Skipping {airfoil_name}: control points missing or length mismatch")
        n_skipped += 1
        continue

    dy_vec = (y_ctrl - y_base).astype(float)
    airfoil_number = airfoil_name.split("_")[-1]
    file_xfoil = os.path.join(xfoil_dir, f"polar_XFOIL_{airfoil_number}_Re{int(Re):.0f}.txt")
    file_nf = os.path.join(nf_dir, f"polar_NeuralFoil_{airfoil_number}_Re{int(Re):.0f}.txt")
    if not (os.path.isfile(file_xfoil) and os.path.isfile(file_nf)):
        print(f"Skipping {airfoil_name}: missing polar files")
        n_skipped += 1
        continue

    alpha_x, cl_x, cd_x, cm_x, _ = read_polar(file_xfoil)
    alpha_nf, cl_nf, cd_nf, cm_nf, _ = read_polar(file_nf)
    err_cl, err_cd, err_cm = compute_errors(alpha_x, cl_x, cd_x, cm_x,
                                            alpha_nf, cl_nf, cd_nf, cm_nf)

    for a, ecl, ecd, ecm in zip(alpha_x, err_cl, err_cd, err_cm):
        a_key = float(a)
        aoa_data[a_key]["dy"].append(dy_vec.copy())
        aoa_data[a_key]["ecl"].append(float(ecl))
        aoa_data[a_key]["ecd"].append(float(ecd))
        aoa_data[a_key]["ecm"].append(float(ecm))
        aoa_data[a_key]["airfoil"].append(airfoil_name)

print(f"\nData collection complete. Files processed: {len(geom_files)}; skipped: {n_skipped}")
print(f"Unique AoA values found: {len(aoa_data)}")

# ============================================================
# === MODEL TRAINING: polynomial + standardized RidgeCV ===
# ============================================================

correction_models = {}
use_sklearn = True
try:
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import Pipeline
    import joblib
except Exception:
    use_sklearn = False

for a in sorted(aoa_data.keys()):
    dy_list = aoa_data[a]["dy"]
    ecl_list = aoa_data[a]["ecl"]
    ecd_list = aoa_data[a]["ecd"]
    ecm_list = aoa_data[a]["ecm"]

    X = np.vstack(dy_list)
    n_samples, n_ctrl = X.shape

    if use_sklearn and n_samples >= 3:
        pipeline_cl = Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-6, 2, 25), cv=min(5, n_samples), scoring="r2"))
        ])
        pipeline_cd = Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-6, 2, 25), cv=min(5, n_samples), scoring="r2"))
        ])
        pipeline_cm = Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-6, 2, 25), cv=min(5, n_samples), scoring="r2"))
        ])

        pipeline_cl.fit(X, ecl_list)
        pipeline_cd.fit(X, ecd_list)
        pipeline_cm.fit(X, ecm_list)

        correction_models[a] = {
            "Cl": pipeline_cl,
            "Cd": pipeline_cd,
            "Cm": pipeline_cm,
            "meta": {"n_samples": n_samples, "method": "RidgeCV"}
        }

        # save models
        try:
            joblib.dump(pipeline_cl, os.path.join(model_dir, f"ridge_cl_aoa_{a:.3f}.joblib"))
            joblib.dump(pipeline_cd, os.path.join(model_dir, f"ridge_cd_aoa_{a:.3f}.joblib"))
            joblib.dump(pipeline_cm, os.path.join(model_dir, f"ridge_cm_aoa_{a:.3f}.joblib"))
        except Exception:
            pass
    else:
        X_design = np.hstack([np.ones((n_samples, 1)), X])
        coeffs_cl, _, _, _ = np.linalg.lstsq(X_design, np.array(ecl_list), rcond=None)
        coeffs_cd, _, _, _ = np.linalg.lstsq(X_design, np.array(ecd_list), rcond=None)
        coeffs_cm, _, _, _ = np.linalg.lstsq(X_design, np.array(ecm_list), rcond=None)

        correction_models[a] = {
            "Cl": (float(coeffs_cl[0]), coeffs_cl[1:].astype(float)),
            "Cd": (float(coeffs_cd[0]), coeffs_cd[1:].astype(float)),
            "Cm": (float(coeffs_cm[0]), coeffs_cm[1:].astype(float)),
            "meta": {"n_samples": n_samples, "method": "OLS"}
        }

    print(f"AoA {a:6.3f}°: samples={n_samples}, stored model.")

# ============================================================
# === SAVE MODEL SUMMARY CSV =================================
# ============================================================

summary_rows = []
for a in sorted(correction_models.keys()):
    meta = correction_models[a]["meta"]
    summary_rows.append({
        "AoA": a,
        "n_samples": meta.get("n_samples", np.nan),
        "method": meta.get("method", "RidgeCV")
    })
pd.DataFrame(summary_rows).to_csv(os.path.join(model_dir, "model_summary.csv"), index=False)
print(f"\n✅ Trained per-AoA models saved into {model_dir}")
print("Model summary written to model_summary.csv")