import numpy as np


class SimpleSTING:
    def __init__(self, grid_size=10, threshold=5):
        self.grid_size = grid_size  # Grid size
        self.threshold = threshold  # Density threshold
        self.grids = {}

    def fit(self, data):
        # 1. Count the points in each cell (n)
        for point in data:
            # Map coordinates to cell indices, e.g. [1.2, 2.5] -> (1, 2)
            grid_idx = tuple((point // self.grid_size).astype(int))
            if grid_idx not in self.grids:
                self.grids[grid_idx] = {'n': 0, 'points': []}
            self.grids[grid_idx]['n'] += 1
            self.grids[grid_idx]['points'].append(point)

        # 2. Drop the sub-threshold noise cells
        self.active_grids = {k: v for k, v in self.grids.items() if v['n'] >= self.threshold}

        # 3. Next: merge adjacent active_grids by BFS/DFS to form clusters
        return self.active_grids


# Usage example
data = np.random.rand(1000, 2) * 100
sting = SimpleSTING(grid_size=5, threshold=10)
result = sting.fit(data)
print(f"活跃网格数量: {len(result)}")