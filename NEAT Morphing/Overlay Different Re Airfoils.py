import os
import numpy as np
import matplotlib.pyplot as plt

# ================================
# USER SETTINGS
# ================================

base_dir = "BestGenomes"
AoA = 5.0
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
# PLOT SETUP
# ================================

plt.figure(figsize=(10,6))

colors = plt.cm.viridis(np.linspace(0,1,len(re_folders)))

found_any = False

# ================================
# LOAD AND PLOT AIRFOILS
# ================================

for color, re_folder in zip(colors, re_folders):

    airfoil_path = os.path.join(
        base_dir,
        re_folder,
        aoa_folder,
        "NEAT_airfoil.dat"
    )

    if not os.path.exists(airfoil_path):
        print(f"Skipping {re_folder} (no airfoil found)")
        continue

    data = np.loadtxt(airfoil_path, skiprows=1)

    x = data[:,0]
    y = data[:,1]

    plt.plot(x, y, lw=2, color=color, label=re_folder)

    found_any = True

if not found_any:
    raise RuntimeError("No airfoils found for that AoA.")

# ================================
# PLOT FORMAT
# ================================

plt.axis("equal")
plt.grid(True)

plt.xlabel("x / chord")
plt.ylabel("y / chord")

plt.title(f"Optimized Airfoils Overlay @ AoA = {AoA}°")

plt.legend(title="Reynolds Number")

plt.tight_layout()
plt.show()