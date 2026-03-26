import os

# ============================================================
# === CONFIGURATION ==========================================
# ============================================================

results_dir = "Simulation Results 5000Re5e4"
repaneled_dir = "Repaneled Geometry"
repaneled_base = "airfoil_xfoil_repaneled_"


# ============================================================
# === MAIN CLEANUP LOOP ======================================
# ============================================================

for polar_file in os.listdir(results_dir):

    if not polar_file.startswith("polar_XFOIL_") or not polar_file.endswith(".txt"):
        continue

    polar_path = os.path.join(results_dir, polar_file)

    with open(polar_path, "r") as f:
        lines = f.readlines()

    # Find the dashed separator line
    data_start = None
    for i, line in enumerate(lines):
        if "------" in line:
            data_start = i + 1
            break

    if data_start is None:
        print(f"⚠️ Could not find separator in {polar_file}")
        continue

    # Extract actual numeric data lines
    data_lines = []
    for l in lines[data_start:]:
        l = l.strip()
        if not l:
            continue

        parts = l.split()

        # Check if first column is numeric (alpha)
        try:
            float(parts[0])
            data_lines.append(l)
        except:
            continue

    # Remove polar if it has 0 or 1 valid data lines
    if len(data_lines) <= 1:

        try:
            airfoil_number = polar_file.split("_")[2]
        except IndexError:
            print(f"⚠️ Cannot parse airfoil number from {polar_file}")
            continue

        repaneled_file = os.path.join(
            repaneled_dir,
            f"{repaneled_base}{airfoil_number}.dat"
        )

        if os.path.exists(repaneled_file):
            os.remove(repaneled_file)
            print(f"🗑️ Removed repaneled airfoil: {repaneled_file}")

        os.remove(polar_path)
        print(f"🗑️ Removed bad polar (≤1 line): {polar_path}")

print("\n✅ Cleanup complete!")