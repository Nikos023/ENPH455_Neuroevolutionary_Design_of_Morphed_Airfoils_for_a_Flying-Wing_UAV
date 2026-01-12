import numpy as np
import neuralfoil as nf
import os
import joblib
import matplotlib.pyplot as plt

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

geom_dir = "../Morphing/Geometry/"
Re = 1e6
airfoil_number = "572"   # change as needed

model_dir = "../Comparison/Comparison Results/global_model"

# ============================================================
# === LOAD GLOBAL CORRECTION MODELS ==========================
# ============================================================

model_cl = joblib.load(os.path.join(model_dir, "global_cl_gb_1000_samples.joblib"))
model_cd = joblib.load(os.path.join(model_dir, "global_cd_gb_1000_samples.joblib"))
model_cm = joblib.load(os.path.join(model_dir, "global_cm_gb_1000_samples.joblib"))

# ============================================================
# === FILE READING ===========================================
# ============================================================

def read_airfoil_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    x_ctrl, y_ctrl = [], []
    xu, yu, xl, yl = [], [], [], []
    section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "=== Control Points" in line:
            section = "ctrl"
            continue
        elif "=== Upper Surface" in line:
            section = "upper"
            continue
        elif "=== Lower Surface" in line:
            section = "lower"
            continue
        elif "===" in line:
            section = None
            continue

        parts = [p.strip() for p in line.split(",")]
        if section == "ctrl":
            x_ctrl.append(float(parts[0]))
            y_ctrl.append(float(parts[1]))
        elif section == "upper":
            xu.append(float(parts[0]))
            yu.append(float(parts[1]))
        elif section == "lower":
            xl.append(float(parts[0]))
            yl.append(float(parts[1]))

    return (
        np.array(x_ctrl), np.array(y_ctrl),
        np.array(xu), np.array(yu),
        np.array(xl), np.array(yl)
    )


def prepare_coordinates_for_neuralfoil(xu, yu, xl, yl):
    coords_upper = np.vstack([xu[::-1], yu[::-1]]).T
    coords_lower = np.vstack([xl, yl]).T
    return np.vstack([coords_upper, coords_lower[1:]])


# ============================================================
# === CORRECTION FEATURE CONSTRUCTION ========================
# ============================================================

def build_base_features(x_ctrl, y_ctrl, x_base, y_base):
    dy_vec = (y_ctrl - y_base).astype(float)
    dy_cumsum = np.cumsum(dy_vec)
    dy_dx = np.gradient(dy_vec, x_ctrl)
    d2y_dx2 = np.gradient(dy_dx, x_ctrl)

    return np.hstack([dy_vec, dy_cumsum, dy_dx, d2y_dx2])


# ============================================================
# === MAIN ===================================================
# ============================================================

def main():
    geom_file = os.path.join(geom_dir, f"airfoil_points_{airfoil_number}.txt")
    base_geom_file = os.path.join(geom_dir, "airfoil_points_000.txt")

    # Load geometry
    x_ctrl, y_ctrl, xu, yu, xl, yl = read_airfoil_file(geom_file)
    x_base, y_base, *_ = read_airfoil_file(base_geom_file)

    base_features = build_base_features(x_ctrl, y_ctrl, x_base, y_base)

    coords = prepare_coordinates_for_neuralfoil(xu, yu, xl, yl)

    # AoA sweep
    alphas = np.linspace(-5, 12, 200)

    print("🧠 Running NeuralFoil...")
    aero = nf.get_aero_from_coordinates(
        coordinates=coords,
        alpha=alphas,
        Re=Re,
        model_size="xxxlarge"
    )

    CL_nf = aero["CL"]
    CD_nf = aero["CD"]
    Cm_nf = aero["CM"]

    # ========================================================
    # === APPLY GLOBAL CORRECTION =============================
    # ========================================================

    CL_corr, CD_corr, Cm_corr = [], [], []

    for i, a in enumerate(alphas):
        X_input = np.hstack([base_features, a]).reshape(1, -1)

        err_cl = model_cl.predict(X_input)[0]
        err_cd = model_cd.predict(X_input)[0]
        err_cm = model_cm.predict(X_input)[0]

        CL_corr.append(CL_nf[i] - err_cl)
        CD_corr.append(CD_nf[i] - err_cd)
        Cm_corr.append(Cm_nf[i] - err_cm)

    CL_corr = np.array(CL_corr)
    CD_corr = np.array(CD_corr)
    Cm_corr = np.array(Cm_corr)

    # ========================================================
    # === PLOTTING (THESIS STYLE, CORRECTED ONLY) ============
    # ========================================================

    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axs = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle(
        f"Corrected NeuralFoil Aerodynamics — Airfoil {airfoil_number} (Re={int(Re):.0f})",
        fontsize=14,
        weight='bold'
    )

    # CL
    axs[0, 0].plot(alphas, CL_corr, 'd-', linewidth=2, label="Corrected NeuralFoil")
    axs[0, 0].set_ylabel("Cl")
    axs[0, 0].set_title("Lift Coefficient")
    axs[0, 0].legend()

    axs[0, 1].plot(alphas, CL_corr - CL_nf, '-', color='tab:green')
    axs[0, 1].axhline(0, color='k', lw=0.8)
    axs[0, 1].set_ylabel("ΔCl")
    axs[0, 1].set_title("Applied Cl Correction")

    # CD
    axs[1, 0].plot(alphas, CD_corr, 'd-', linewidth=2, label="Corrected NeuralFoil")
    axs[1, 0].set_ylabel("Cd")
    axs[1, 0].set_title("Drag Coefficient")
    axs[1, 0].legend()

    axs[1, 1].plot(alphas, CD_corr - CD_nf, '-', color='tab:green')
    axs[1, 1].axhline(0, color='k', lw=0.8)
    axs[1, 1].set_ylabel("ΔCd")
    axs[1, 1].set_title("Applied Cd Correction")

    # CM
    axs[2, 0].plot(alphas, Cm_corr, 'd-', linewidth=2, label="Corrected NeuralFoil")
    axs[2, 0].set_ylabel("Cm")
    axs[2, 0].set_xlabel("AoA (deg)")
    axs[2, 0].set_title("Pitching Moment Coefficient")
    axs[2, 0].legend()

    axs[2, 1].plot(alphas, Cm_corr - Cm_nf, '-', color='tab:green')
    axs[2, 1].axhline(0, color='k', lw=0.8)
    axs[2, 1].set_ylabel("ΔCm")
    axs[2, 1].set_xlabel("AoA (deg)")
    axs[2, 1].set_title("Applied Cm Correction")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================
# === RUN ====================================================
# ============================================================

if __name__ == "__main__":
    main()