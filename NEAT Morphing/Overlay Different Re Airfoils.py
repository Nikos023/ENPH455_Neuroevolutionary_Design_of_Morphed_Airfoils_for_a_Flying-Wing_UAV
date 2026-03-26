#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ================================
# USER SETTINGS
# ================================
base_dir = "BestGenomes"
AoA = 12.00
aoa_folder = f"{AoA:.2f} Degrees"

# ================================
# FIND ALL REYNOLDS FOLDERS
# ================================
re_folders = [
    f for f in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("Re")
]
re_folders.sort()

# ================================
# EXTRACT REYNOLDS NUMBERS
# ================================
re_values = []
for f in re_folders:
    try:
        re_val = float(f.replace("Re", "").replace("_", ""))
    except:
        re_val = np.nan
    re_values.append(re_val)

re_values = np.array(re_values)

# ================================
# PLOT AIRFOILS WITH COLORBAR
# ================================
plt.figure(figsize=(10,6))

norm = mpl.colors.Normalize(vmin=np.min(re_values), vmax=np.max(re_values))
cmap = plt.cm.viridis
colors = cmap(norm(re_values))

found_any = False
aoa_vals_for_color = []

# Sort by Reynolds number
sorted_pairs = sorted(zip(re_values, re_folders, colors), key=lambda x: x[0])

for _, re_folder, color in sorted_pairs:
    airfoil_path = os.path.join(base_dir, re_folder, aoa_folder, "NEAT_airfoil.dat")
    if not os.path.exists(airfoil_path):
        print(f"Skipping {re_folder} (no airfoil found)")
        continue

    data = np.loadtxt(airfoil_path, skiprows=1)
    x, y = data[:,0], data[:,1]

    plt.plot(x, y, lw=2, color=color, label=re_folder)
    found_any = True
    aoa_vals_for_color.append(AoA)

if not found_any:
    raise RuntimeError("No airfoils found for that AoA.")

plt.axis("equal")
plt.grid(True)
plt.xlabel("x / chord")
plt.ylabel("y / chord")
plt.title(f"Optimized Airfoils Overlay @ AoA = {AoA}°")

# Colorbar based on Reynolds number
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
cbar.set_label("Reynolds Number", fontweight='bold')

plt.legend(title="Reynolds Number")
plt.tight_layout()
plt.show()

# ================================
# SUBPLOT: AIRFOIL + CP
# ================================
cp_data = []
geom_data = []

for re_folder in re_folders:
    airfoil_path = os.path.join(base_dir, re_folder, aoa_folder, "NEAT_airfoil.dat")
    cp_path = os.path.join(base_dir, re_folder, aoa_folder, "cp.dat")
    if not os.path.exists(airfoil_path):
        continue
    data = np.loadtxt(airfoil_path, skiprows=1)
    x, y = data[:,0], data[:,1]
    geom_data.append((re_folder, x, y))

    if os.path.exists(cp_path):
        data_cp = np.loadtxt(cp_path, skiprows=3)
        x_cp, cp = data_cp[:,0], data_cp[:,1]
        cp_data.append((re_folder, x_cp, cp))

if len(geom_data)==0 and len(cp_data)==0:
    raise RuntimeError("No geometry or Cp data found for subplot.")

# --- Subplot figure ---
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex=False, gridspec_kw={'height_ratios':[1,1]})

# Compute Reynolds values for geom_data
geom_re_values = []
for re_folder, _, _ in geom_data:
    val = float(re_folder.replace("Re", "").replace("_", ""))
    geom_re_values.append(val)

geom_re_values = np.array(geom_re_values)

# Use SAME cmap and norm as first plot
colors = cmap(norm(geom_re_values))

cp_re_vals = [float(r.replace("Re","").replace("_","")) for r,_,_ in cp_data]
sorted_cp = sorted(zip(cp_re_vals, cp_data), key=lambda x: x[0])

for re_val, (re_folder, x_cp, cp) in sorted_cp:
    color = cmap(norm(re_val))
    le = np.argmin(x_cp)
    ax1.plot(x_cp[:le+1], cp[:le+1], color=color)
    ax1.plot(x_cp[le:], cp[le:], color=color, linestyle='--')

ax1.invert_yaxis()
ax1.set_ylabel("Cp", fontweight='bold')
ax1.set_title(f"Cp(x) Overlay @ AoA = {AoA}°", fontweight='bold')
ax1.grid(True)

# Geometry plot (bottom)
geom_re_vals = [float(r.replace("Re","").replace("_","")) for r,_,_ in geom_data]
sorted_geom = sorted(zip(geom_re_vals, geom_data), key=lambda x: x[0])

for re_val, (re_folder, x, y) in sorted_geom:
    color = cmap(norm(re_val))
    ax2.plot(x, y, color=color)

ax2.set_aspect('equal', 'box')
ax2.set_xlabel("x/c", fontweight='bold')
ax2.set_ylabel("y/c", fontweight='bold')
ax2.set_title("Airfoil Geometry Overlay", fontweight='bold')
ax2.grid(True)
ax2.set_ylim(-0.1, 0.15)

# Colorbar
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([0.92, 0.155, 0.025, 0.725])
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Reynolds Number", fontweight='bold')

plt.show()