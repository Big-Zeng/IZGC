from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

R = 30.0
THETA = 5
DELTA_G = 2.46
MAX_BUFFER_DISTANCE = 0
NOISE_LEVELS = tuple(round(0.05 * i, 2) for i in range(11))

SPIRAL_CSV = Path("../data/spiral.csv")
RAW_RESULTS_CSV = Path("../results_noise_spiral/noise_experiment_raw.csv")
OUTPUT_DIR = Path("paper_figures/izgc_noise_assignments")

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def normalize_coordinates(X: np.ndarray) -> np.ndarray:
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    xr = x_max - x_min
    yr = y_max - y_min
    out = np.zeros_like(X, dtype=float)
    out[:, 0] = ((X[:, 0] - x_min) / xr * 100) if xr > 0 else 0
    out[:, 1] = ((X[:, 1] - y_min) / yr * 100) if yr > 0 else 0
    return out

def load_spiral_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    X = df[["x", "y"]].astype(float).to_numpy()
    y = df["target"].to_numpy()
    return normalize_coordinates(X), y

def add_noise_to_dataset(X, y_true, noise_ratio, seed):
    if noise_ratio <= 0:
        return X.copy(), y_true.copy()
    rng = np.random.default_rng(seed)
    n_clean = len(X)
    n_noise = int(round(noise_ratio * n_clean / max(1e-12, 1.0 - noise_ratio)))
    mins = X.min(axis=0) - 4
    maxs = X.max(axis=0) + 4
    noise = rng.uniform(mins, maxs, size=(n_noise, 2))
    X_noisy = np.vstack([X, noise])
    y_noisy = np.concatenate([y_true, np.full(n_noise, -1, dtype=int)])
    return X_noisy, y_noisy

def run_izgc(X: np.ndarray) -> np.ndarray:
    import ContourBasedClusteringWithoutZ22 as ContourBasedClusteringWithoutZ
    points = np.asarray(X)
    tree = cKDTree(points)
    density = np.array([len(tree.query_ball_point(p, R)) for p in points])
    combined_data = [(*point, density[i], i) for i, point in enumerate(points)]
    model = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
        combined_data,
        grid_size=DELTA_G,
        max_buffer_distance=MAX_BUFFER_DISTANCE,
        min_cluster_size=THETA,
        interactive=False,
        merge_small_clusters=False,
    )
    model.perform_clustering()
    return np.asarray(model.get_predicted_labels(), dtype=int)

def select_median_run(raw: pd.DataFrame, noise_ratio: float):
    sub = raw[
        (raw["Algorithm"] == "IZGC")
        & np.isclose(raw["noise_ratio"].astype(float), noise_ratio)
    ].copy()
    if sub.empty:
        raise ValueError(f"No IZGC results for noise={noise_ratio:.0%}")
    med = sub["Inlier_ARI"].median()
    idx = (sub["Inlier_ARI"] - med).abs().idxmin()
    return sub.loc[idx]

def plot_one_noise_level(X_base, y_base, raw, noise_ratio):
    chosen = select_median_run(raw, noise_ratio)
    seed = int(chosen["Seed"])
    X, _ = add_noise_to_dataset(X_base, y_base, noise_ratio, seed)
    y_pred = run_izgc(X)
    pred_k = len([lab for lab in np.unique(y_pred) if lab != -1])
    ari = float(chosen["Inlier_ARI"])

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for lab in np.unique(y_pred):
        m = y_pred == lab
        if lab == -1:
            ax.scatter(X[m, 0], X[m, 1], s=15, alpha=0.35,
                       c="gray", marker="x", linewidths=0.7)
        else:
            ax.scatter(X[m, 0], X[m, 1], s=14, alpha=0.75,
                       edgecolors="none")

    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 104)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X", fontsize=15)
    ax.set_ylabel("Y", fontsize=15)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="both", labelsize=15, length=3)
    # ax.set_title(f"Noise={noise_ratio:.0%},  K={pred_k}",
    #              fontsize=12, pad=7)
    fig.tight_layout()

    pct = int(round(noise_ratio * 100))
    base = OUTPUT_DIR / f"izgc_noise_{pct:02d}"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print(f"Noise={noise_ratio:.0%}: seed={seed}, Inlier_ARI={ari:.4f}, K={pred_k}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SPIRAL_CSV.exists():
        raise FileNotFoundError(f"Cannot find {SPIRAL_CSV}")
    if not RAW_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {RAW_RESULTS_CSV}. Run noise_spiral_revised.py first."
        )

    X_base, y_base = load_spiral_csv(SPIRAL_CSV)
    raw = pd.read_csv(RAW_RESULTS_CSV)

    required = {"Algorithm", "noise_ratio", "Seed", "Inlier_ARI"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    for noise_ratio in NOISE_LEVELS:
        plot_one_noise_level(X_base, y_base, raw, noise_ratio)

    print(f"\nSaved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
