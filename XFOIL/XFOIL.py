import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

xfoil_path = "/Users/nicholasburen/Downloads/xfoil/bin/xfoil"
airfoil_base_path = "../Morphing/Geometry/airfoil_points_"  # Base path without number and extension
airfoil_ext = ".dat"

repaneled_dir = "Repaneled Geometry"
os.makedirs(repaneled_dir, exist_ok=True)

results_dir = "Simulation Results"
os.makedirs(results_dir, exist_ok=True)

repaneled_base = "airfoil_xfoil_repaneled_"  # Base name for repaneled files

# Flow and solver parameters
Re = 1e6
Mach = 0.0
iter_limit = 300
Ncrit = 9
panel_count = 300  # Recommended: 150–400 for small UAVs

# Angle of attack sweep (deg)
alpha_points = np.linspace(-5, 12, 200)

# ============================================================
# === UTILITY FUNCTIONS ======================================
# ============================================================

def read_airfoil_coords(filename):
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
    i_le = np.argmin(coords[:, 0])
    upper = coords[:i_le + 1]
    lower = coords[i_le:]
    return upper[:, 0], upper[:, 1], lower[:, 0], lower[:, 1]

def safe_interp(x_src, y_src, x_target):
    order = np.argsort(x_src)
    xs = x_src[order]
    ys = y_src[order]
    xs_unique, idx = np.unique(np.round(xs, 12), return_index=True)
    ys_unique = ys[idx]
    return np.interp(x_target, xs_unique, ys_unique)

def repanel_airfoil(xu, yu, xl, yl, n_points=200):
    n_half = n_points // 2 + 1
    beta = np.linspace(0, np.pi, n_half)
    x_cos = 0.5 * (1 - np.cos(beta))
    y_upper = safe_interp(xu, yu, x_cos)
    y_lower = safe_interp(xl, yl, x_cos)
    x_upper = x_cos[::-1]
    y_upper = y_upper[::-1]
    x_lower = x_cos[1:]
    y_lower = y_lower[1:]
    x_all = np.concatenate([x_upper, x_lower])
    y_all = np.concatenate([y_upper, y_lower])
    return x_all, y_all

# ============================================================
# === MAIN LOOP FOR FILES 0-100 =============================
# ============================================================

for i in range(1001):
    # Construct file paths
    num_str = f"{i:03d}"  # zero-padded number
    airfoil_file = f"{airfoil_base_path}{num_str}{airfoil_ext}"
    repaneled_file = os.path.join(repaneled_dir, f"{repaneled_base}{num_str}.dat")
    polar_file = os.path.join(results_dir, f"polar_XFOIL_{num_str}_Re{int(Re):.0f}.txt")

    if not os.path.exists(airfoil_file):
        print(f"⚠️ Skipping {airfoil_file} — file not found.")
        continue

    print(f"\n=== Processing {airfoil_file} ===")

    # Repanel
    xu, yu, xl, yl = read_airfoil_coords(airfoil_file)
    x_all, y_all = repanel_airfoil(xu, yu, xl, yl, panel_count)
    with open(repaneled_file, "w") as f:
        f.write("Repaneled_Airfoil\n")
        for xi, yi in zip(x_all, y_all):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    # Build XFOIL commands
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

    # Run XFOIL
    process = subprocess.run([xfoil_path], input=xfoil_input, text=True, capture_output=True)
    if process.returncode != 0:
        print("❌ XFOIL failed to run.")
        continue

    print(f"✅ Polar saved: {polar_file}")

print("\n🎯 All files processed!")