import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

xfoil_path = "/Users/nicholasburen/Downloads/xfoil/bin/xfoil"
airfoil_file = "../Morphing/Geometry/airfoil_points_015.dat"  # Original airfoil geometry
repaneled_file = "airfoil_xfoil_repaneled_last_tested.dat"

# Flow and solver parameters
Re = 1e6
Mach = 0.0
iter_limit = 300
Ncrit = 9
panel_count = 300  # Recommended: 150–400 for small UAVs

# Create simulation results directory if it doesn't exist
results_dir = "Simulation Results"
os.makedirs(results_dir, exist_ok=True)

# Extract the numeric part from the airfoil filename (e.g., 'airfoil_points_000.dat')
airfoil_number = os.path.splitext(os.path.basename(airfoil_file))[0].split('_')[-1]

# Automatically set polar filename based on airfoil number and Re
polar_file = os.path.join(results_dir, f"polar_XFOIL_{airfoil_number}_Re{int(Re):.0f}.txt")

# Angle of attack sweep (deg)
alpha_points = np.linspace(-5, 12, 200)

# ============================================================
# === UTILITY FUNCTIONS ======================================
# ============================================================

def read_airfoil_coords(filename):
    """Read x,y coords from .dat file, skipping header text."""
    raw = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x, y = map(float, parts[:2])
                raw.append((x, y))
            except ValueError:
                continue
    coords = np.array(raw)
    if coords.shape[0] < 10:
        raise RuntimeError(f"File {filename} has too few numeric points.")
    # Split at the leading edge (min x)
    i_le = np.argmin(coords[:, 0])
    upper = coords[:i_le + 1]
    lower = coords[i_le:]
    return upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1]


def safe_interp(x_src, y_src, x_target):
    """Safe interpolation ensuring monotonic, unique x."""
    order = np.argsort(x_src)
    xs = x_src[order]
    ys = y_src[order]
    xs_unique, idx = np.unique(np.round(xs, 12), return_index=True)
    ys_unique = ys[idx]
    return np.interp(x_target, xs_unique, ys_unique)


def repanel_airfoil(xu, yu, xl, yl, n_points=200):
    """Repanel airfoil using cosine spacing (better LE/TE resolution)."""
    n_half = n_points // 2 + 1
    beta = np.linspace(0, np.pi, n_half)
    x_cos = 0.5 * (1 - np.cos(beta))  # cosine spacing from 0→1

    y_upper = safe_interp(xu, yu, x_cos)
    y_lower = safe_interp(xl, yl, x_cos)

    x_upper = x_cos[::-1]      # TE -> LE
    y_upper = y_upper[::-1]
    x_lower = x_cos[1:]        # LE -> TE
    y_lower = y_lower[1:]

    x_all = np.concatenate([x_upper, x_lower])
    y_all = np.concatenate([y_upper, y_lower])
    return x_all, y_all

# ============================================================
# === RE-PANEL AIRFOIL =======================================
# ============================================================

xu, yu, xl, yl = read_airfoil_coords(airfoil_file)
x_all, y_all = repanel_airfoil(xu, yu, xl, yl, panel_count)

with open(repaneled_file, "w") as f:
    f.write("Repaneled_Airfoil\n")
    for xi, yi in zip(x_all, y_all):
        f.write(f"{xi:.6f} {yi:.6f}\n")

print(f"✅ Repaneled airfoil saved as {repaneled_file} ({len(x_all)} points)")

# ============================================================
# === BUILD XFOIL COMMANDS ===================================
# ============================================================

if os.path.exists(polar_file):
    os.remove(polar_file)

commands = [
    f"LOAD {repaneled_file}",
    "OPER",
    "VPAR",
    f"N {Ncrit}",
    "",  # exit paneling
    f"VISC {Re:.0f}",
    f"MACH {Mach}",
    f"ITER {iter_limit}",
    "PACC",
    polar_file,
    "",  # end PACC prompt
]

for alpha in alpha_points:
    commands.append(f"ALFA {alpha:.4f}")

commands += ["CACC", "QUIT"]

xfoil_input = "\n".join(commands) + "\n"

# ============================================================
# === RUN XFOIL ==============================================
# ============================================================

print(f"🚀 Running XFOIL | Re = {Re:.0e} | Ncrit = {Ncrit} | {len(alpha_points)} AoA points")

process = subprocess.run([xfoil_path], input=xfoil_input, text=True, capture_output=True)

if process.returncode != 0:
    print("❌ XFOIL failed to run.")
print("\n--- XFOIL OUTPUT (first 100 lines) ---")
print("\n".join(process.stdout.splitlines()[:100]))
print("--------------------------------------\n")

# ============================================================
# === LOAD POLAR DATA ========================================
# ============================================================

if not os.path.exists(polar_file):
    raise FileNotFoundError(f"❌ Polar file '{polar_file}' was not created. Check for convergence issues.")

try:
    data = np.loadtxt(polar_file, skiprows=12)
    if data.ndim == 1:
        data = data.reshape(1, -1)
except Exception as e:
    raise RuntimeError(f"Failed to read polar file: {e}")

alpha = data[:, 0]
cl = data[:, 1]
cd = data[:, 2]
cm = data[:, 4]

# ============================================================
# === PLOT RESULTS ===========================================
# ============================================================

fig = plt.figure(figsize=(10, 8))
fig.suptitle(f"XFOIL Aerodynamic Characteristics (Re={Re:.0e}, Ncrit={Ncrit})", fontsize=14)

gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

# Lift curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(alpha, cl, '-o', markersize=3)
ax1.set_xlabel("Angle of Attack (deg)")
ax1.set_ylabel("Lift Coefficient (Cl)")
ax1.set_title("Lift Curve")
ax1.grid(True)

# Drag curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(alpha, cd, '-o', markersize=3)
ax2.set_xlabel("Angle of Attack (deg)")
ax2.set_ylabel("Drag Coefficient (Cd)")
ax2.set_title("Drag Polar")
ax2.grid(True)

# Pitching moment
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(alpha, cm, '-o', color='tab:red', markersize=3)
ax3.set_xlabel("Angle of Attack (deg)")
ax3.set_ylabel("Pitching Moment Coefficient (Cm)")
ax3.set_title("Pitching Moment vs. Angle of Attack")
ax3.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("✅ Analysis complete and plots generated.")