import matplotlib.pyplot as plt

# Reynolds numbers
Re_list = [1e6, 5e5, 2e5, 1e5, 5e4]

# Error reduction (%) from your printed results
improvements_cl  = [68.99, 62.53, 59.62, 61.94, 45.94]
improvements_cd  = [71.00, 66.61, 64.43, 70.30, 55.82]
improvements_cm  = [79.98, 74.90, 65.91, 61.64, 50.83]
improvements_all = [71.60, 65.46, 60.97, 62.63, 47.34]

# Plot
plt.figure(figsize=(10,6))
plt.plot(Re_list, improvements_cl,  marker='o', linestyle='-', label="Cl")
plt.plot(Re_list, improvements_cd,  marker='s', linestyle='-', label="Cd")
plt.plot(Re_list, improvements_cm,  marker='^', linestyle='-', label="Cm")
plt.plot(Re_list, improvements_all, marker='d', linestyle='-', label="All Combined")

plt.xscale("log")
plt.xlabel("Reynolds Number", fontweight='bold')
plt.ylabel("Error Reduction (%)", fontweight='bold')
plt.title("Global ML Correction Error Reduction vs Reynolds Number", fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()