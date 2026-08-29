"""
IZGC overlap experiment: six distances, six independent figures.

Extracted from plot_izgc_overlap_vary_d.py.
Only IZGC is retained.

Conditions:
    d = 6, 5, 4, 3, 2, 1
    sigma = 1.0
    n_per_cluster = 500

Each condition is generated and plotted separately.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import norm


# ============================================================
# Project import
# ============================================================

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ContourBasedClusteringWithoutZ22 as CBC


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 20260825

D_VALUES = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
SIGMA_FIXED = 1.0
N_PER_CLUSTER = 500

# Same fixed IZGC parameters for all six overlap conditions
R = 20.0
THETA = 10
DELTA_G = 1.4
MAX_BUFFER_DISTANCE = 0

OUTPUT_DIR = Path("../paper_figures/izgc_overlap_assignments")


# ============================================================
# Paper style
# ============================================================

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

CLUSTER_COLORS = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#9b59b6",
]


# ============================================================
# Data generation
# ============================================================

def normalize_coordinates(X: np.ndarray) -> np.ndarray:
    """Same min-max normalization as the original experiment."""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    Xn = np.zeros_like(X, dtype=float)

    Xn[:, 0] = (
        (X[:, 0] - x_min) / x_range * 100
        if x_range > 0 else 0
    )

    Xn[:, 1] = (
        (X[:, 1] - y_min) / y_range * 100
        if y_range > 0 else 0
    )

    return Xn


def compute_overlap_ratio(d: float, sigma: float) -> float:
    """Overlap = 2 Phi[-d/(2 sigma)]."""
    if sigma <= 0:
        return 0.0

    return float(
        2 * norm.cdf(
            -d / (2 * sigma)
        )
    )


def generate_overlapping_blobs(
    seed: int,
    d: float,
    sigma: float = 1.0,
    n_per_cluster: int = 500,
):
    """Generate the same two Gaussian clusters as the original experiment."""
    rng = np.random.default_rng(seed)

    mu1 = np.array([
        -d / 2.0,
        0.0,
    ])

    mu2 = np.array([
        d / 2.0,
        0.0,
    ])

    cov = np.array([
        [sigma**2, 0.0],
        [0.0, sigma**2],
    ])

    X1 = rng.multivariate_normal(
        mu1,
        cov,
        n_per_cluster,
    )

    X2 = rng.multivariate_normal(
        mu2,
        cov,
        n_per_cluster,
    )

    X = np.vstack([
        X1,
        X2,
    ])

    y_true = np.r_[
        np.zeros(n_per_cluster, dtype=int),
        np.ones(n_per_cluster, dtype=int),
    ]

    return normalize_coordinates(X), y_true


# ============================================================
# IZGC
# ============================================================

def run_izgc(X: np.ndarray) -> np.ndarray:
    """Same IZGC call as the supplied overlap plotting code."""
    points = np.asarray(X)

    tree = cKDTree(points)

    density = np.array([
        len(tree.query_ball_point(p, R))
        for p in points
    ])

    combined = [
        (*point, density[i], i)
        for i, point in enumerate(points)
    ]

    model = CBC.ContourBasedClusteringWithoutZ(
        combined,
        grid_size=DELTA_G,
        max_buffer_distance=MAX_BUFFER_DISTANCE,
        min_cluster_size=THETA,
        interactive=False,
        merge_small_clusters=True,
    )

    model.perform_clustering()

    return np.asarray(
        model.get_predicted_labels(),
        dtype=int,
    )


# ============================================================
# One d -> one figure
# ============================================================

def plot_one_distance(d: float) -> None:

    X, _ = generate_overlapping_blobs(
        seed=RANDOM_SEED,
        d=d,
        sigma=SIGMA_FIXED,
        n_per_cluster=N_PER_CLUSTER,
    )

    overlap = compute_overlap_ratio(
        d,
        SIGMA_FIXED,
    )

    y_pred = run_izgc(X)

    pred_labels = sorted([
        lab
        for lab in np.unique(y_pred)
        if lab != -1
    ])

    pred_k = len(pred_labels)

    # --------------------------------------------------------
    # Independent square figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(4.5, 4.5)
    )

    for i, lab in enumerate(pred_labels):

        mask = (
            y_pred == lab
        )

        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=14,
            alpha=0.75,
            c=CLUSTER_COLORS[
                i % len(CLUSTER_COLORS)
            ],
            edgecolors="none",
        )

    # Predicted noise, if any
    noise_mask = (
        y_pred == -1
    )

    if np.any(noise_mask):
        ax.scatter(
            X[noise_mask, 0],
            X[noise_mask, 1],
            s=15,
            alpha=0.30,
            c="gray",
            marker="x",
            linewidths=0.8,
        )

    # --------------------------------------------------------
    # Same axes for all six figures
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        100,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel(
        "X",
        fontsize=15,
    )

    ax.set_ylabel(
        "Y",
        fontsize=15,
    )

    ax.set_xticks([
        0, 25, 50, 75, 100
    ])

    ax.set_yticks([
        0, 25, 50, 75, 100
    ])

    ax.tick_params(
        axis="both",
        labelsize=15,
        length=3,
    )

    # No sigma in the title.
    # Show d, overlap, and predicted K only.
    ax.set_title(
        rf"$d={d:.0f}$, Overlap={overlap:.2%}, "
        rf"$K_{{\mathrm{{pred}}}}={pred_k}$",
        fontsize=15,
        pad=7,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save one independent figure
    # --------------------------------------------------------

    d_tag = int(round(d))

    base = (
        OUTPUT_DIR
        / f"izgc_overlap_d{d_tag}"
    )

    # Vector PDF
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    # 600 dpi PNG
    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )

    # SVG
    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"d={d:.0f}: "
        f"overlap={overlap:.4%}, "
        f"K_pred={pred_k}"
    )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Generating six separate IZGC "
        "overlap figures...\n"
    )

    for d in D_VALUES:
        plot_one_distance(d)

    print(
        f"\nSaved to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
