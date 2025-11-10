import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import os

# ============================================================
# === FUNCTIONS ==============================================
# ============================================================

def thickness_distribution(x, t):
    """NACA 4-digit half-thickness distribution (yt)."""
    return 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

def compute_airfoil(x, yc, yt):
    """Computes upper and lower surface coordinates."""
    dyc_dx = np.gradient(yc, x)
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    return xu, yu, xl, yl

def smooth_camber(x_ctrl, y_ctrl, x_dense):
    """Smooths camber line with cubic spline + Gaussian filter."""
    spline = make_interp_spline(x_ctrl, y_ctrl, k=3)
    yc_dense = spline(x_dense)
    yc_dense = gaussian_filter1d(yc_dense, sigma=1.2)
    return yc_dense

# ============================================================
# === BASE PARAMETERS ========================================
# ============================================================

m, p, t = 0.02, 0.4, 0.12
num_points = 1000
num_ctrl = 10

# Cosine spacing for x
beta = np.linspace(0, np.pi, num_points)
x = (1 - np.cos(beta)) / 2

# Base camber and thickness
yc_base = np.where(
    x < p,
    m / p**2 * (2 * p * x - x**2),
    m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2),
)
yt_base = thickness_distribution(x, t)

# Control points (same logic as in interactive version)
n_each_side = num_ctrl // 2
x_ctrl_left = np.linspace(0, 1/3, n_each_side, endpoint=False)
x_ctrl_right = np.linspace(2/3, 1, n_each_side)
x_ctrl = np.concatenate([x_ctrl_left, x_ctrl_right])
y_ctrl_base = np.interp(x_ctrl, x, yc_base)

# ============================================================
# === RANDOM SAMPLE GENERATION ===============================
# ============================================================

os.makedirs("Geometry", exist_ok=True)
n_samples = 30

for i in range(1, n_samples + 1):
    # ---- Create smooth random offsets (like manual morphs) ----
    noise_strength = np.random.uniform(0.01, 0.06)  # typical offset magnitude
    random_offsets = np.random.normal(0, noise_strength, size=num_ctrl)
    random_offsets = gaussian_filter1d(random_offsets, sigma=1.5)

    # Apply offsets to control points
    y_ctrl = y_ctrl_base + random_offsets

    # Smooth camber line
    yc = smooth_camber(x_ctrl, y_ctrl, x)

    # Compute airfoil surfaces
    xu, yu, xl, yl = compute_airfoil(x, yc, yt_base)

    # ---- Save files ----
    base_name = f"airfoil_points_{i:03d}"
    txt_filename = f"Geometry/{base_name}.txt"
    dat_filename = f"Geometry/{base_name}.dat"

    # Save .txt file (same as interactive version)
    with open(txt_filename, "w") as f:
        f.write("=== Airfoil Parameters ===\n")
        f.write(f"m = {m}\n")
        f.write(f"p = {p}\n")
        f.write(f"t = {t}\n\n")
        f.write("=== Control Points ===\n")
        for xi, yi, off in zip(x_ctrl, y_ctrl, random_offsets):
            f.write(f"{xi:.5f}, {yi:.5f}, {off:.5f}\n")
        f.write("\n=== Upper Surface ===\n")
        for xi, yi in zip(xu, yu):
            f.write(f"{xi:.5f}, {yi:.5f}\n")
        f.write("\n=== Lower Surface ===\n")
        for xi, yi in zip(xl, yl):
            f.write(f"{xi:.5f}, {yi:.5f}\n")

    # Save .dat file for XFOIL
    N = 100
    beta = np.linspace(0, np.pi, N)
    x_cos = 0.5 * (1 - np.cos(beta))
    y_upper_interp = np.interp(x_cos, xu, yu)
    y_lower_interp = np.interp(x_cos, xl, yl)
    x_all = np.concatenate([x_cos[::-1], x_cos[1:]])
    y_all = np.concatenate([y_upper_interp[::-1], y_lower_interp[1:]])

    with open(dat_filename, "w") as f:
        f.write(f"{base_name}\n")
        for xi, yi in zip(x_all, y_all):
            f.write(f"{xi:.6f} {yi:.6f}\n")

    print(f"✅ Saved {txt_filename} and {dat_filename}")

print("\n🎉 Done! 30 randomized airfoil geometries generated in /Geometry/")