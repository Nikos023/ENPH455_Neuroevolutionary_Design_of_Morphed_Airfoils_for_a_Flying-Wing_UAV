import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# ============================================================
# === FILE READING FUNCTION ==================================
# ============================================================

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
# === FILE PATHS =============================================
# ============================================================

file_xfoil = "../XFOIL/Simulation Results/polar_XFOIL_015_Re1000000.txt"
file_nf = "../Neuralfoil/Simulation Results/polar_NeuralFoil_015_Re1000000.txt"
Re = 1e6  # Reynolds number used for naming

# ============================================================
# === LOAD DATA ==============================================
# ============================================================

alpha_x, cl_x, cd_x, cm_x, _ = read_polar(file_xfoil)
alpha_nf, cl_nf, cd_nf, cm_nf, _ = read_polar(file_nf)

print(f"Loaded {len(alpha_x)} XFOIL points | {len(alpha_nf)} NeuralFoil points")

# ============================================================
# === INTERPOLATE NEURALFOIL TO MATCH XFOIL ALPHAS ===========
# ============================================================

f_cl = interp1d(alpha_nf, cl_nf, fill_value="extrapolate")
f_cd = interp1d(alpha_nf, cd_nf, fill_value="extrapolate")
f_cm = interp1d(alpha_nf, cm_nf, fill_value="extrapolate")

cl_nf_i = f_cl(alpha_x)
cd_nf_i = f_cd(alpha_x)
cm_nf_i = f_cm(alpha_x)

# ============================================================
# === COMPUTE ERRORS =========================================
# ============================================================

err_cl = cl_nf_i - cl_x
err_cd = cd_nf_i - cd_x
err_cm = cm_nf_i - cm_x

pct_err_cl = 100 * err_cl / np.maximum(np.abs(cl_x), 1e-8)
pct_err_cd = 100 * err_cd / np.maximum(np.abs(cd_x), 1e-8)
pct_err_cm = 100 * err_cm / np.maximum(np.abs(cm_x), 1e-8)

# ============================================================
# === PRINT NUMERIC ERROR SUMMARY ===========================
# ============================================================

mean_abs_cl = np.mean(np.abs(err_cl))
mean_abs_cd = np.mean(np.abs(err_cd))
mean_abs_cm = np.mean(np.abs(err_cm))

rms_pct_cl = np.sqrt(np.mean(pct_err_cl**2))
rms_pct_cd = np.sqrt(np.mean(pct_err_cd**2))
rms_pct_cm = np.sqrt(np.mean(pct_err_cm**2))

summary = f"""
=== NeuralFoil vs XFOIL Error Summary ===
Mean |ΔCl| = {mean_abs_cl:.4f}
Mean |ΔCd| = {mean_abs_cd:.5f}
Mean |ΔCm| = {mean_abs_cm:.4f}
RMS % Error Cl = {rms_pct_cl:.2f}%
RMS % Error Cd = {rms_pct_cd:.2f}%
RMS % Error Cm = {rms_pct_cm:.2f}%
"""

print(summary)

# ============================================================
# === SAVE DETAILED ERROR REPORT =============================
# ============================================================

# Create comparison results directory if it doesn't exist
comparison_dir = "Comparison Results"
os.makedirs(comparison_dir, exist_ok=True)

# Extract airfoil number from XFOIL filename (assuming format 'airfoil_XFOIL_000_ReXXXXXXX.txt')
airfoil_number = os.path.splitext(os.path.basename(file_xfoil))[0].split('_')[2]

# Automatically set error filename based on airfoil number and Re
output_error_file = os.path.join(comparison_dir, f"comparison_error_{airfoil_number}_Re{int(Re):.0f}.txt")

# Save the error report
with open(output_error_file, "w") as f:
    f.write("==============================================================\n")
    f.write(" NeuralFoil vs XFOIL Comparison Report\n")
    f.write("==============================================================\n\n")
    f.write(summary + "\n")
    f.write("AoA(deg)     Cl_XFOIL   Cl_NF   ΔCl     Cd_XFOIL   Cd_NF   ΔCd     Cm_XFOIL   Cm_NF   ΔCm\n")
    f.write("--------------------------------------------------------------------------------------\n")
    for i in range(len(alpha_x)):
        f.write(f"{alpha_x[i]:8.3f}  {cl_x[i]:9.5f}  {cl_nf_i[i]:9.5f}  {err_cl[i]:8.5f}  "
                f"{cd_x[i]:9.5f}  {cd_nf_i[i]:9.5f}  {err_cd[i]:8.5f}  "
                f"{cm_x[i]:9.5f}  {cm_nf_i[i]:9.5f}  {err_cm[i]:9.5f}\n")

print(f"💾 Error report saved to '{output_error_file}'")

# ============================================================
# === PLOTTING ===============================================
# ============================================================

plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(3, 2, figsize=(13, 10))
fig.suptitle(f"XFOIL vs NeuralFoil Comparison — Airfoil {airfoil_number} (Re={int(Re):.0f})",
             fontsize=14, weight='bold')

# --- CL ---
axs[0, 0].plot(alpha_x, cl_x, 'o-', label='XFOIL')
axs[0, 0].plot(alpha_nf, cl_nf, 's--', label='NeuralFoil')
axs[0, 0].set_ylabel("Cl")
axs[0, 0].legend()
axs[0, 0].set_title("Lift Coefficient")

axs[0, 1].plot(alpha_x, err_cl, 'r-')
axs[0, 1].axhline(0, color='k', lw=0.8)
axs[0, 1].set_ylabel("ΔCl (NF - XFOIL)")
axs[0, 1].set_title("Cl Error vs AoA")

# --- CD ---
axs[1, 0].plot(alpha_x, cd_x, 'o-', label='XFOIL')
axs[1, 0].plot(alpha_nf, cd_nf, 's--', label='NeuralFoil')
axs[1, 0].set_ylabel("Cd")
axs[1, 0].legend()
axs[1, 0].set_title("Drag Coefficient")

axs[1, 1].plot(alpha_x, err_cd, 'r-')
axs[1, 1].axhline(0, color='k', lw=0.8)
axs[1, 1].set_ylabel("ΔCd (NF - XFOIL)")
axs[1, 1].set_title("Cd Error vs AoA")

# --- CM ---
axs[2, 0].plot(alpha_x, cm_x, 'o-', label='XFOIL')
axs[2, 0].plot(alpha_nf, cm_nf, 's--', label='NeuralFoil')
axs[2, 0].set_ylabel("Cm")
axs[2, 0].legend()
axs[2, 0].set_title("Pitching Moment Coefficient")
axs[2, 0].set_xlabel("Angle of Attack (deg)")

axs[2, 1].plot(alpha_x, err_cm, 'r-')
axs[2, 1].axhline(0, color='k', lw=0.8)
axs[2, 1].set_ylabel("ΔCm (NF - XFOIL)")
axs[2, 1].set_title("Cm Error vs AoA")
axs[2, 1].set_xlabel("Angle of Attack (deg)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()