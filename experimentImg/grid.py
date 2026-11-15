import matplotlib.pyplot as plt
import numpy as np


def plot_marching_squares_states_v2():
    # Vertex coordinates (x, y): v0 bottom-left, v1 bottom-right, v2 top-right, v3 top-left
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # Edge midpoint, used to draw the contour segment
    # Edge index: 0 bottom, 1 right, 2 top, 3 left
    midpoints = np.array([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]])

    # Lookup table for the 16 cases (LUT)
    # Define the edge pairs to connect for each case index
    lut = [
        [],  # 0: 0000
        [[3, 0]],  # 1: 0001
        [[0, 1]],  # 2: 0010
        [[3, 1]],  # 3: 0011
        [[1, 2]],  # 4: 0100
        [[3, 0], [1, 2]],  # 5: 0101 (saddle point)
        [[0, 2]],  # 6: 0110
        [[3, 2]],  # 7: 0111
        [[2, 3]],  # 8: 1000
        [[2, 0]],  # 9: 1001
        [[0, 1], [2, 3]],  # 10: 1010 (saddle point)
        [[2, 1]],  # 11: 1011
        [[1, 3]],  # 12: 1100
        [[1, 0]],  # 13: 1101
        [[0, 3]],  # 14: 1110
        []  # 15: 1111
    ]

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(16):
        ax = axes[i]

        # 1. Draw the cell edges
        square = plt.Polygon(vertices, fill=None, edgecolor='gray', linestyle='--', lw=1)
        ax.add_patch(square)

        # 2. Read the binary state (0 or 1) of the four vertices for case i
        # Bit extraction: bit0 -> v0, bit1 -> v1, bit2 -> v2, bit3 -> v3
        vertex_states = [(i >> bit) & 1 for bit in range(4)]

        # 3. Draw the vertices
        for v_idx, state in enumerate(vertex_states):
            color = 'black' if state == 1 else 'white'
            ax.plot(vertices[v_idx, 0], vertices[v_idx, 1], 'o',
                    markersize=14, markerfacecolor=color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)

        # 4. Draw the contour segments
        lines_to_draw = lut[i]
        for edge_pair in lines_to_draw:
            start_p = midpoints[edge_pair[0]]
            end_p = midpoints[edge_pair[1]]
            ax.plot([start_p[0], end_p[0]], [start_p[1], end_p[1]],
                    color='#d62728', linewidth=3.5, zorder=4)

        # 5. Set the coordinates and display
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # 6. Show index i below each subplot
        ax.text(0.5, -0.15, f"Index {i}", transform=ax.transAxes,
                ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('marching_squares_16_states.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_marching_squares_states_v2()