#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# === LOAD TXT FORMAT ========================================
# ============================================================

def load_txt_airfoil(filename):
    xu, yu, xl, yl = [], [], [], []
    mode = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if "Upper Surface" in line:
                mode = "upper"
                continue
            elif "Lower Surface" in line:
                mode = "lower"
                continue
            elif line.startswith("===") or line == "":
                continue

            if mode == "upper":
                x, y = map(float, line.split(","))
                xu.append(x)
                yu.append(y)
            elif mode == "lower":
                x, y = map(float, line.split(","))
                xl.append(x)
                yl.append(y)

    return np.array(xu), np.array(yu), np.array(xl), np.array(yl)


# ============================================================
# === LOAD DAT FORMAT ========================================
# ============================================================

def load_dat_airfoil(filename):
    data = np.loadtxt(filename, skiprows=1)
    x, y = data[:, 0], data[:, 1]

    # Split upper/lower automatically
    mid = np.argmin(x)  # leading edge
    xu, yu = x[:mid+1], y[:mid+1]
    xl, yl = x[mid:], y[mid:]

    return xu, yu, xl, yl


# ============================================================
# === PLOT FUNCTION ==========================================
# ============================================================

def plot_airfoil(filename):
    if filename.endswith(".txt"):
        xu, yu, xl, yl = load_txt_airfoil(filename)
    elif filename.endswith(".dat"):
        xu, yu, xl, yl = load_dat_airfoil(filename)
    else:
        raise ValueError("Unsupported file type. Use .txt or .dat")

    plt.figure(figsize=(10, 4))
    plt.plot(xu, yu, linewidth=2, label="Upper Surface")
    plt.plot(xl, yl, linewidth=2, label="Lower Surface")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Airfoil: {filename}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# === MAIN ===================================================
# ============================================================

if __name__ == "__main__":
    # 🔥 Just change this line
    file_path = "Geometry/airfoil_points_0001.dat"
    # file_path = "Geometry(Not)/airfoil_points_001.dat"

    plot_airfoil(file_path)