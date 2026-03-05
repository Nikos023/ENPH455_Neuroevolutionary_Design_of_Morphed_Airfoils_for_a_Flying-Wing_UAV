import os
import numpy as np

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

results_dir = "Simulation Results 5000Re3e5"  # where polar files are stored
repaneled_dir = "Repaneled Geometry"           # where repaneled airfoils are stored
repaneled_base = "airfoil_xfoil_repaneled_"    # base name for repaneled files

# Expected number of alpha points
alpha_points = np.linspace(-5, 12, 200)
expected_count = len(alpha_points)

# ============================================================
# === MAIN CLEANUP LOOP ======================================
# ============================================================

for polar_file in os.listdir(results_dir):
    if not polar_file.startswith("polar_XFOIL_") or not polar_file.endswith(".txt"):
        continue

    polar_path = os.path.join(results_dir, polar_file)
    # Read file and count number of angle entries (skip header lines)
    with open(polar_path, "r") as f:
        lines = f.readlines()

    # Skip empty lines and header lines that contain "alpha" or non-numeric data
    data_lines = [line for line in lines if line.strip() and all(c.isdigit() or c in ".-eE +\t" for c in line.split()[0])]

    if len(data_lines) < expected_count:
        # Extract airfoil number from polar file name
        try:
            num_str = polar_file.split("_")[2]
        except IndexError:
            print(f"⚠️ Cannot parse airfoil number from {polar_file}")
            continue

        # Delete repaneled airfoil
        repaneled_file = os.path.join(repaneled_dir, f"{repaneled_base}{num_str}.dat")
        if os.path.exists(repaneled_file):
            os.remove(repaneled_file)
            print(f"🗑️ Removed incomplete airfoil: {repaneled_file}")

        # Optionally, remove polar file too
        os.remove(polar_path)
        print(f"🗑️ Removed incomplete polar: {polar_path}")

print("\n✅ Cleanup complete!")