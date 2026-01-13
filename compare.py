import numpy as np


class SimpleSTING:
    def __init__(self, grid_size=10, threshold=5):
        self.grid_size = grid_size  # 网格大小
        self.threshold = threshold  # 密度阈值
        self.grids = {}

    def fit(self, data):
        # 1. 统计每个网格内的点数 (n)
        for point in data:
            # 将坐标映射到网格索引 (e.g., [1.2, 2.5] -> (1, 2))
            grid_idx = tuple((point // self.grid_size).astype(int))
            if grid_idx not in self.grids:
                self.grids[grid_idx] = {'n': 0, 'points': []}
            self.grids[grid_idx]['n'] += 1
            self.grids[grid_idx]['points'].append(point)

        # 2. 过滤掉不达标的“噪声”网格
        self.active_grids = {k: v for k, v in self.grids.items() if v['n'] >= self.threshold}

        # 3. 后续步骤：你可以通过 BFS/DFS 算法合并相邻的 active_grids 形成簇
        return self.active_grids


# 使用示例
data = np.random.rand(1000, 2) * 100
sting = SimpleSTING(grid_size=5, threshold=10)
result = sting.fit(data)
print(f"活跃网格数量: {len(result)}")