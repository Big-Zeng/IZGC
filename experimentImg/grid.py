import matplotlib.pyplot as plt
import numpy as np


def plot_marching_squares_states_v2():
    # 顶点坐标 (x, y): v0:左下, v1:右下, v2:右上, v3:左上
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    # 边中点坐标，用于绘制等值线段
    # 边索引: 0:底边, 1:右边, 2:顶边, 3:左边
    midpoints = np.array([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]])

    # 16种状态的查找表 (LUT)
    # 定义每种索引下需要连接的边对
    lut = [
        [],  # 0: 0000
        [[3, 0]],  # 1: 0001
        [[0, 1]],  # 2: 0010
        [[3, 1]],  # 3: 0011
        [[1, 2]],  # 4: 0100
        [[3, 0], [1, 2]],  # 5: 0101 (鞍点)
        [[0, 2]],  # 6: 0110
        [[3, 2]],  # 7: 0111
        [[2, 3]],  # 8: 1000
        [[2, 0]],  # 9: 1001
        [[0, 1], [2, 3]],  # 10: 1010 (鞍点)
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

        # 1. 绘制网格单元边框
        square = plt.Polygon(vertices, fill=None, edgecolor='gray', linestyle='--', lw=1)
        ax.add_patch(square)

        # 2. 根据索引 i 提取 4 个顶点的二值状态 (0或1)
        # 位运算提取: bit0->v0, bit1->v1, bit2->v2, bit3->v3
        vertex_states = [(i >> bit) & 1 for bit in range(4)]

        # 3. 绘制顶点
        for v_idx, state in enumerate(vertex_states):
            color = 'black' if state == 1 else 'white'
            ax.plot(vertices[v_idx, 0], vertices[v_idx, 1], 'o',
                    markersize=14, markerfacecolor=color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)

        # 4. 绘制等值线段
        lines_to_draw = lut[i]
        for edge_pair in lines_to_draw:
            start_p = midpoints[edge_pair[0]]
            end_p = midpoints[edge_pair[1]]
            ax.plot([start_p[0], end_p[0]], [start_p[1], end_p[1]],
                    color='#d62728', linewidth=3.5, zorder=4)

        # 5. 设置坐标与显示
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # 6. 在每个小图下方显示 Index i
        ax.text(0.5, -0.15, f"Index {i}", transform=ax.transAxes,
                ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('marching_squares_16_states.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_marching_squares_states_v2()