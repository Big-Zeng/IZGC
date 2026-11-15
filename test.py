import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib
matplotlib.use('TkAgg')
def polygon_concave_vertices(poly):
    concave = []
    n = len(poly)

    for i in range(n):
        p_prev = poly[i-1]
        p_curr = poly[i]
        p_next = poly[(i+1)%n]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross < 0:  # Concave vertex
            concave.append(p_curr)

    return np.array(concave)


# ===== Example: take one path from the contours =====
fig, ax = plt.subplots()

# Placeholder function
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) + np.cos(Y)

cs = ax.contour(X, Y, Z, levels=[0.2])
paths = cs.get_paths()
poly = paths[0].verticesll
# Locate the concave vertex of the polygon
concave = polygon_concave_vertices(poly)

# Show the results
ax.plot(poly[:,0], poly[:,1], '-k')
ax.scatter(concave[:,0], concave[:,1], color='red', s=40)

plt.title("多边形凹点（红色） → 潜在山谷方向")
plt.show()
