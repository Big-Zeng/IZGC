from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 20260825

# Six representative overlap conditions for 2x3 grid
D_VALUES = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

# Fixed in the original experiment
SIGMA_FIXED = 1.0

N_PER_CLUSTER = 500

OUTPUT_DIR = Path("paper_figures")
OUTPUT_BASENAME = "overlap_vary_d_2x3"


# ============================================================
# Plot style
# Keep this consistent with the Spiral Noise 1x3 figure
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": 9,

    # Same as Spiral Noise figure
    "axes.titlesize": 12,
    "axes.labelsize": 12,

    # Keep ticks smaller than labels
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,

    "legend.fontsize": 8.5,

    "axes.linewidth": 0.8,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Same normalization used in the original overlap experiment
# ============================================================

def normalize_coordinates(X: np.ndarray) -> np.ndarray:
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    X_norm = np.zeros_like(X, dtype=float)

    X_norm[:, 0] = (
        ((X[:, 0] - x_min) / x_range * 100)
        if x_range > 0 else 0
    )

    X_norm[:, 1] = (
        ((X[:, 1] - y_min) / y_range * 100)
        if y_range > 0 else 0
    )

    return X_norm


# ============================================================
# Same overlap-rate definition as the original experiment
# ============================================================

def compute_overlap_ratio(d: float, sigma: float) -> float:
    """
    Overlap = 2 * Phi(-d / (2*sigma))
    """
    if sigma <= 0:
        return 0.0

    return float(
        2 * norm.cdf(
            -d / (2 * sigma)
        )
    )


# ============================================================
# Same Gaussian-data generator as the original experiment
# ============================================================

def generate_overlapping_blobs(
    seed: int,
    d: float,
    sigma: float = 1.0,
    n_per_cluster: int = 500,
):

    rng = np.random.default_rng(seed)

    mu1 = np.array([-d / 2.0, 0.0])
    mu2 = np.array([ d / 2.0, 0.0])

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

    X = np.vstack([X1, X2])

    y = np.r_[
        np.zeros(n_per_cluster, dtype=int),
        np.ones(n_per_cluster, dtype=int),
    ]

    return normalize_coordinates(X), y


# ============================================================
# 2 x 3 paper figure
# ============================================================

def plot_overlap_2x3():

    # 2 rows x 3 columns, 4.5 inches per panel x 4.5 inches high
    # Total figure size: 13.5 x 9.0 inches
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 9.0),
    )

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    for ax, d in zip(axes_flat, D_VALUES):

        X, y = generate_overlapping_blobs(
            seed=RANDOM_SEED,
            d=d,
            sigma=SIGMA_FIXED,
            n_per_cluster=N_PER_CLUSTER,
        )

        # Two original Gaussian clusters
        for lab in np.unique(y):

            mask = (y == lab)

            ax.scatter(
                X[mask, 0],
                X[mask, 1],

                # Similar visual weight to the Spiral Noise figure
                s=14,

                alpha=0.75,
                edgecolors="none",
            )

        overlap = compute_overlap_ratio(
            d,
            SIGMA_FIXED,
        )

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        ax.set_title(
            rf"$d={d:.0f}$, Overlap={overlap:.1%}",
            fontsize=15,
            pad=6,
        )

        # ----------------------------------------------------
        # X / Y labels
        # Same font size as Spiral Noise
        # ----------------------------------------------------

        ax.set_xlabel(
            "X",
            fontsize=15,
        )

        ax.set_ylabel(
            "Y",
            fontsize=15,
        )

        # ----------------------------------------------------
        # Coordinate ticks
        # ----------------------------------------------------

        ax.set_xticks(
            [0, 25, 50, 75, 100]
        )

        ax.set_yticks(
            [0, 25, 50, 75, 100]
        )

        ax.tick_params(
            axis="both",
            labelsize=15,
            length=3,
        )

        # Same coordinate system for all panels
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

    # Similar spacing to Spiral Noise figure
    fig.tight_layout(
        pad=0.8,
        w_pad=0.8,
        h_pad=1.0,  # Add some vertical spacing between rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{OUTPUT_BASENAME}.pdf"
    )

    png_path = (
        OUTPUT_DIR
        / f"{OUTPUT_BASENAME}.png"
    )

    svg_path = (
        OUTPUT_DIR
        / f"{OUTPUT_BASENAME}.svg"
    )

    # Vector PDF
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    # 600 dpi raster
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    # Vector SVG
    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}  (600 dpi)")
    print(f"  {svg_path}")


if __name__ == "__main__":
    plot_overlap_2x3()