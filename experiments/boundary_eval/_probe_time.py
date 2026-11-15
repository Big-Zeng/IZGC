"""Quick timing probe for the 0.001 Delta_g grid (stage 1 only)."""
import multiprocessing as mp
import time

import numpy as np


def w(gs):
    import adapters as ad
    import scene_gen as sg
    s = sg.build("gap_4", 9001)
    t = time.time()
    ad.run("IZGC-Full", s["X"], s["k"],
           dict(grid_size=gs, R=30.0, min_size=5, buffer=0),
           scene=s, seed=9001, polygonise=False)
    return gs, time.time() - t


if __name__ == "__main__":
    grid = [round(0.5 + 4.5 * i / 11, 3) for i in range(12)]
    t0 = time.time()
    with mp.Pool(8) as pool:
        for gs, dt in pool.imap_unordered(w, grid):
            print(f"  dg={gs:<6} {dt:5.1f}s", flush=True)
    print(f"wall {time.time() - t0:.1f}s")
