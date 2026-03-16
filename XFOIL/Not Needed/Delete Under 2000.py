import os
import re

# ============================================================
# === DIRECTORY ==============================================
# ============================================================

folder = "Simulation Results 5000Re5e5"

# ============================================================
# === FILE PATTERN ===========================================
# ============================================================

pattern = re.compile(r"polar_XFOIL_(\d+)_Re\d+\.txt")

deleted_count = 0
kept_count = 0

# ============================================================
# === MAIN LOOP ==============================================
# ============================================================

for file in os.listdir(folder):

    match = pattern.match(file)

    if not match:
        continue

    airfoil_number = int(match.group(1))
    filepath = os.path.join(folder, file)

    if airfoil_number < 2001:

        print(f"❌ Deleting {file}")
        os.remove(filepath)
        deleted_count += 1

    else:

        kept_count += 1

# ============================================================
# === SUMMARY ================================================
# ============================================================

print("\n===================================")
print(f"Files deleted : {deleted_count}")
print(f"Files kept    : {kept_count}")
print("===================================")