# Calculate of coefficient for the correlation

import numpy as np
import matplotlib.pyplot as plt

#Import data points from given data

x = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
y = np.array([ 2,  0, -1, -2, -1,-1,-2,-1, 0,-1])

x_mean = np.mean(x)
y_mean = np.mean(y)

#Calculation

numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
r_manual = numerator / denominator

print("Coefficient= ",r_manual)

#Graph display and save

plt.figure(figsize=(6,6))
plt.scatter(x, y, color='blue', s=80)
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.6)

plt.title(f"Coefficient ≈ {r_manual:.3f}")
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig("graph1.png", dpi=300, bbox_inches='tight')
plt.close()


