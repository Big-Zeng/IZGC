import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Fix the seed for reproducible results
np.random.seed(10)

# 1. Generate random point data
# Number of points; 30 keeps the set sparse
num_points = 15
# Generate 2-D random points in [0, 100]
points = np.random.rand(num_points, 2) * 100

# 2. Build the Delaunay triangulation (TIN)
tri = Delaunay(points)

# 3. Create the figure canvas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
# Set the global font (Arial or Times New Roman are recommended)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# --- Fig. 1: randomly distributed point set ---
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(points[:, 0], points[:, 1], color='#1f77b4', s=60, edgecolors='k', linewidth=0.5, zorder=3)
#ax1.set_title("Discrete Sampling Points", fontsize=16)
ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.set_xlabel("X ", fontsize=12)
ax1.set_ylabel("Y ", fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.5)
# Save a 300 DPI figure
plt.savefig('scatter_points_300dpi.png', dpi=300, bbox_inches='tight')

# --- Fig. 2: the resulting TIN ---
fig2, ax2 = plt.subplots(figsize=(6, 6))
# Draw the triangulation edges
ax2.triplot(points[:, 0], points[:, 1], tri.simplices, color='#7f7f7f', linewidth=1.2, zorder=1)
# Draw the vertices
ax2.scatter(points[:, 0], points[:, 1], color='#1f77b4', s=60, edgecolors='k', linewidth=0.5, zorder=3)
#ax2.set_title("Generated TIN (Delaunay)", fontsize=16)
ax2.set_xlim(-5, 105)
ax2.set_ylim(-5, 105)
ax2.set_xlabel("X ", fontsize=12)
ax2.set_ylabel("Y ", fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.5)
# Save a 300 DPI figure
plt.savefig('tin_network_300dpi.png', dpi=300, bbox_inches='tight')


# Show the image
plt.show()