import os
import re

# ============================================================
# === DIRECTORIES ============================================
# ============================================================

neuralfoil_dir = "Simulation Results 5000Re3e5"
xfoil_dir = "../XFOIL/Not Needed/Simulation Results 5000Re1e6"

# ============================================================
# === FILE PATTERN ===========================================
# ============================================================

pattern = re.compile(r"polar_NeuralFoil_(\d{4})_Re(\d+)\.txt")

deleted_count = 0
kept_count = 0

# ============================================================
# === MAIN LOOP ==============================================
# ============================================================

for file in os.listdir(neuralfoil_dir):

    match = pattern.match(file)

    if not match:
        continue

    airfoil_id = match.group(1)
    reynolds = match.group(2)

    neuralfoil_path = os.path.join(neuralfoil_dir, file)

    xfoil_file = f"polar_XFOIL_{airfoil_id}_Re{reynolds}.txt"
    xfoil_path = os.path.join(xfoil_dir, xfoil_file)

    # If XFOIL file does not exist → delete NeuralFoil file
    if not os.path.exists(xfoil_path):

        print(f"❌ Deleting {file} (no XFOIL match)")
        os.remove(neuralfoil_path)
        deleted_count += 1

    else:

        kept_count += 1

# ============================================================
# === SUMMARY ================================================
# ============================================================

print("\n===================================")
print(f"NeuralFoil files kept   : {kept_count}")
print(f"NeuralFoil files deleted: {deleted_count}")
print("===================================")